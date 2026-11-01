function o = rl_obs_v13(theta, theta_dot, phi, phi_dot, theta_ref, u_FLC)
%RL_OBS_V13  Vector quan sat cho lop hieu chinh RL. 6 chieu, da chuan hoa.
%
% ===========================================================================
% MOT NGUON DUY NHAT cho dinh nghia quan sat.
% ---------------------------------------------------------------------------
% Ham nay duoc goi CA trong moi truong huan luyen (rwip_env_v13) LAN trong bo
% mo phong danh gia (sim_rwip_v13, mode PSOFLC_RL). Neu viet hai ban rieng thi
% chung se lech nhau -- da xay ra trong du an nay voi dung sai xac lap
% (thetaTol 0.02 trong ham muc tieu PSO so voi 0.002 trong script bao cao),
% khien PSO toi uu tren mot moc de hon thuc te va tra ve nghiem kem hon ca
% diem xuat phat.
%
% ---------------------------------------------------------------------------
% VI SAO 6 CHIEU NAY
% ---------------------------------------------------------------------------
% Ban cu dung [omega_p, omega_r, u_FLC] -- THIEU ca theta lan phi.
%   * Thieu theta: dac ta rang buoc vot lo va ts_theta, agent khong quan sat
%     duoc dai luong minh phai toi uu.
%   * Thieu phi: NGUY HIEM HON. Dinh luat bao toan rang buoc theta voi phi:
%         Int sin(theta) dt = ( Ir*phidot(T) - bp*theta0 ) / mgl_eq
%     Agent khong thay phi se khong phan biet duoc "dua banh da ve nghi" voi
%     "de banh da quay tu do" -- chinh la che do hong ma SMC-ESO ban goc roi
%     vao (do duoc RMSEphi = 10.02 rad, IAEphi = 523).
%   * theta_ref: de agent phan biet "con lac lech do nhieu" voi "con lac lech
%     do vong ngoai CHU DONG nghieng de keo banh da ve".
%   * u_FLC: kien truc residual -- agent can biet no dang hieu chinh cai gi.
%
% He so chuan hoa lay theo BIEN DO TU NHIEN tren bao lam viec da do, khong
% theo dung sai xac lap (bai hoc tu loi trong so phan thuong: chuan hoa phi
% theo phi_tol=0.02 lam so hang phi ap dao 876 lan so hang theta).
% ===========================================================================

% ===========================================================================
% S_PH: 2.0 -> 10.0  (SUA THEO SO DO, ngay 08/08)
% ---------------------------------------------------------------------------
% Gia tri cu 2.0 di kem chu thich "phi len toi ~2 khi co nhieu lon" -- mot uoc
% luong khong dan nguon do. Do lai bang Test_Env_RewardDecomp_v13 tren DUNG
% phan bo kich ban huan luyen (p_kick = 1.0, kick_amp 0.5-3.0 rad/s):
%
%     max|phi| tren bo nen = 23.985 rad   (~3.8 vong tron)
%
% Tuc phi/S_PH cham ~12, trong khi NAM thanh phan quan sat con lai deu duoc
% chuan hoa ve O(1) dung theo nguyen tac ghi ngay tren ("he so chuan hoa lay
% theo BIEN DO TU NHIEN tren bao lam viec DA DO"). Mot dac trung dau vao lon
% gap 12 lan cac dac trung khac lam hong dieu kien so cua lop dau tien trong
% mang 256-256 ReLU -- anh huong CA actor (qua BC) LAN critic.
%
% Day la ung vien DOC LAP cho trieu chung "critic khong hoc" (Q0 ~ 0 so voi
% gia tri that), khong lien quan gi den DiscountFactor hay ham thuong. No ton
% tai TRUOC moi ban va da lam.
%
% 10.0 dat max|phi| chuan hoa ve ~2.4 va gia tri dien hinh ve ~1. Khong chon
% dung 24 vi nhu vay se ep phan lon quy dao xuong ~0.04 -- mat do phan giai o
% dai hay gap. Kiem tra lai bang muc [3c] cua Test_Env_RewardDecomp_v13.
%
% CANH BAO PHAM VI: file nay dung CHUNG cho huan luyen (rwip_env_v13) va danh
% gia (sim_rwip_v13 mode PSOFLC_RL) -- do la ly do no ton tai. Doi o day la
% doi CA HAI cung luc, dung y do. Nhung moi agent da luu TRUOC thay doi nay
% deu hoc tren thang cu va KHONG con dung: phai chay lai BC. Train_RL_v13_bc
% sinh lai du lieu BC o buoc [1/5]-[2/5] moi lan chay nen dieu nay tu dong.
% ===========================================================================
S_TH   = 0.15;    % rad     -- theta trong bao lam viec
S_THD  = 3.0;     % rad/s
S_PH   = 10.0;    % rad     -- DO DUOC: max|phi| = 23.985 tren kich ban nhieu
S_PHD  = 20.0;    % rad/s
S_TREF = 0.035;   % rad     -- bang dung theta_ref_max
S_U    = 24.0;    % V       -- bao hoa dien ap

o = [ theta     / S_TH   ;
      theta_dot / S_THD  ;
      phi       / S_PH   ;
      phi_dot   / S_PHD  ;
      theta_ref / S_TREF ;
      u_FLC     / S_U    ];

% ---------------------------------------------------------------------------
% CANH BAO MOT LAN neu mot he so chuan hoa lech thang do. Lan truoc S_PH = 2.0
% sai 12 lan va khong ai biet trong suot nhieu lan chay -- vi khong co gi bao.
% Nguong 5 rong rai (chuan hoa dung cho ~1, cham 2-3 la binh thuong); vuot 5
% nghia la mot he so duoi day khong con khop bao lam viec that.
% persistent -> chi in mot lan moi phien MATLAB, khong lam ngap 4000 buoc/ep.
% ---------------------------------------------------------------------------
persistent warned
if isempty(warned), warned = false; end
if ~warned && any(abs(o) > 5)
    nm = {'theta','theta_dot','phi','phi_dot','theta_ref','u_FLC'};
    sc = [S_TH S_THD S_PH S_PHD S_TREF S_U];
    k  = find(abs(o) > 5);
    warned = true;
    for kk = k(:)'
        fprintf(2, ['(!) rl_obs_v13: thanh phan "%s" chuan hoa = %.1f (ky vong ~1).\n' ...
                    '    He so dang dung = %g nhung gia tri tho = %g. He so nay co the\n' ...
                    '    khong con khop bao lam viec that -- xem muc [3c] cua\n' ...
                    '    Test_Env_RewardDecomp_v13. (Chi canh bao mot lan moi phien.)\n'], ...
                nm{kk}, o(kk), sc(kk), o(kk)*sc(kk));
    end
end
end
