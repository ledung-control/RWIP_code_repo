"""Trich xuat trong so actor tu file .mat cua rlTD3Agent (MAT v7, MCOS opaque).

Kien truc (Train_RL_v13_bc.m dong 326-333):
    featureInput(6) -> fc1(256) -> relu -> fc2(256) -> relu -> fc3(1) -> tanh

Khoi actor phan biet voi hai critic o cho: actor co DUNG MOT ma tran (256,256),
critic co HAI (do duong obs va duong action gop lai).
"""
import struct
import numpy as np
import scipy.io as sio

NPT = {7: '<f4', 9: '<f8'}


def _scan(buf):
    """Quet tuyen tinh, tra ve [(offset, dims, array)] cho moi mang so."""
    out, n, o = [], len(buf), 0
    while o + 24 <= n:
        typ, sz = struct.unpack('<II', buf[o:o + 8])
        if typ == 14 and 16 < sz < n - o:
            p = o + 8
            t, s = struct.unpack('<II', buf[p:p + 8])
            if t == 6 and s == 8:
                cls = np.frombuffer(buf[p + 8:p + 16], '<u4')[0] & 0xFF
                p2 = p + 16
                t2, s2 = struct.unpack('<II', buf[p2:p2 + 8])
                if t2 == 5 and s2 in (8, 12) and cls in (6, 7):
                    dims = tuple(int(x) for x in np.frombuffer(buf[p2 + 8:p2 + 8 + s2], '<i4'))
                    p3 = p2 + 8 + ((s2 + 7) // 8) * 8
                    t3, s3 = struct.unpack('<II', buf[p3:p3 + 8])
                    p4 = p3 + 8 + ((s3 + 7) // 8) * 8 if (t3 == 1 and s3 > 0) else (
                        p3 + 8 if (t3 & 0xFFFF) == 1 else p3)
                    tag, = struct.unpack('<I', buf[p4:p4 + 4])
                    if (tag >> 16) & 0xFFFF:          # small data element
                        t4, s4, dstart = tag & 0xFFFF, (tag >> 16) & 0xFFFF, p4 + 4
                    else:
                        t4, s4 = struct.unpack('<II', buf[p4:p4 + 8])
                        dstart = p4 + 8
                    npn = int(np.prod(dims))
                    if t4 in NPT and npn > 0 and s4 == npn * np.dtype(NPT[t4]).itemsize:
                        a = np.frombuffer(buf[dstart:dstart + s4], NPT[t4]).reshape(dims, order='F')
                        out.append((o, dims, a.astype(np.float64), int(cls)))
        o += 8
    return out


def load_actor(path):
    """Tra ve dict(W1,b1,W2,b2,W3,b3, aBias_file) cho mot file agent."""
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    raw = np.ascontiguousarray(d['__function_workspace__']).tobytes()
    arrs = _scan(raw)
    big = [(o, dm, a) for o, dm, a, c in arrs if int(np.prod(dm)) >= 256]

    # gom khoi: moi khoi bat dau tai mot ma tran (256,6)
    starts = [i for i, (o, dm, a) in enumerate(big) if dm == (256, 6)]
    blocks = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(big)
        blocks.append(big[i:j])
    # actor = khoi co DUNG MOT ma tran (256,256)
    cand = [b for b in blocks if sum(1 for _, dm, _ in b if dm == (256, 256)) == 1]
    if len(cand) != 1:
        raise RuntimeError(f'{path}: tim thay {len(cand)} khoi ung vien actor')
    blk = cand[0]
    W1 = next(a for _, dm, a in blk if dm == (256, 6))
    W2 = next(a for _, dm, a in blk if dm == (256, 256))
    W3 = next(a for _, dm, a in blk if dm == (1, 256)).reshape(1, 256)
    b_cands = [(o, a.ravel()) for o, dm, a in blk if dm == (256, 1)]
    o_W2 = next(o for o, dm, a in blk if dm == (256, 256))
    o_W3 = next(o for o, dm, a in blk if dm == (1, 256))
    b1 = next(a for o, a in b_cands if o < o_W2)
    b2 = next(a for o, a in b_cands if o_W2 < o < o_W3)

    # b3 = mang (1,1) dau tien co offset lon hon W3 (thu tu bang Learnables)
    b3 = 0.0
    for o, dm, a, c in arrs:
        if o > o_W3 and int(np.prod(dm)) == 1 and c == 7:
            b3 = float(a.ravel()[0])
            break
    aBias_file = float(d['aBias']) if 'aBias' in d else None
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=np.array([b3]),
                aBias_file=aBias_file)


def pi(net, obs):
    """Chinh sach tat dinh: a = tanh(W3 relu(W2 relu(W1 o + b1) + b2) + b3)."""
    o = np.asarray(obs, dtype=np.float64).reshape(6)
    h = np.maximum(net['W1'] @ o + net['b1'], 0.0)
    h = np.maximum(net['W2'] @ h + net['b2'], 0.0)
    return float(np.tanh(net['W3'] @ h + net['b3'])[0])
