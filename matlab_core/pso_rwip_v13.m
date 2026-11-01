function pso_rwip_v13(mode)
%   pso_rwip_v13              -> chay toi uu day du (40-90 phut)
%   pso_rwip_v13('precheck')  -> CHI kiem tra ham muc tieu (~2 phut), khong toi uu
%
% LUON chay 'precheck' truoc. No kiem 4 dieu da tung sai trong du an nay:
%   (1) chi so CHET  -- so hang trong cost khong doi giua cac nghiem khac nhau
%                       (da gap: OS_theta = max|theta| luon = theta0)
%   (2) LECH DINH NGHIA -- cost va script bao cao do khac nhau
%                       (da gap: thetaTol 0.02 vs 0.002, phiTol 0.10 vs 0.020)
%   (3) MIEN KHA THI  -- dac ta co dat duoc boi it nhat mot nghiem khong
%   (4) MOC SO SANH   -- J cua particle gieo, de biet PSO co cai thien that khong
if nargin < 1, mode = 'run'; end
%PSO_RWIP_V13  Toi uu PSO cho RWIP PSO-FLC -- ban da sua loi cost cua v12.
%
% ===========================================================================
% BA THAY DOI CAN BAN SO VOI pso_rwip.m (v12)
% ---------------------------------------------------------------------------
% (A) BO WRAP PHI -- day la loi nghiem trong nhat cua v12.
%     v12 dong 64/491:   useWrapPhiForCost = true
%                        phi_cost = mod(phi + pi, 2*pi) - pi
%     Wrap goc BANH DA ve [-pi,pi]. Quy dao phi = -76 rad (12 vong quay) sau
%     wrap chi con -0.602 rad. Nen ca 5 so hang phat phi (IAE_phi, OS_phi,
%     wEnd_phi, wSlope_phi, settle_pen) deu dang do mot song rang cua = NHIEU.
%     PSO khong nhung khong thay loi, no con CHU DONG chon toc do quay sao cho
%     phi wrap ve gan 0 dung luc t=30 s. Do la reward hacking dien hinh.
%     -> v13: phi KHONG BAO GIO duoc wrap. Wrap chi dung cho goc con lac.
%
% (B) KHONG GIAN TIM KIEM DOI TU "vi tri MF" SANG "tham so vat ly".
%     v12 toi uu 15 toa do MF tu do -> PSO tao ra FIS lech tam
%     (FLC(0,0)=+4.6 V) va bang luat khong doi xung. Ngoai ra 15 chieu MF
%     phan lon la thua vi bai toan doi xung qua goc 0.
%     v13 dung 10 chieu, tat ca DEU co y nghia vat ly va DEU dam bao
%     FLC(0,0)=0 theo cau truc:
%        1  s_e     do rong MF input1  (doi xung)      [0.15 0.95]
%        2  s_de    do rong MF input2  (doi xung)      [0.15 0.95]
%        3  s_u     do rong MF output  (doi xung)      [0.15 0.95]
%        4  Ke      chuan hoa e_theta                  [0.5  6.0]
%        5  Kde     chuan hoa de_theta                 [0.05 1.0]
%        6  Ku      bien do dien ap FLC (V)            [4    24]
%        7  Kd_theta                                   [0    4]
%        8  Kd_phi                                     [0    14]
%        9  Kp_phi  vong ngoai, ty le phi              [0    0.05]
%       10  Kv_phi  vong ngoai, ty le phi_dot          [0    0.30]
%     FIS duoc DUNG LAI TU DAU moi lan danh gia -> luon hop le, luon doi xung,
%     khong can toan bo chuoi repair/sanitize/clamp phuc tap cua v12.
%
% (C) FIS truyen qua BIEN WORKSPACE, khong qua FILE.
%     v12 ghi de fis file moi lan eval -> voi parfor la race condition.
%     v13: set_param(fuzzyBlk,'FIS','fisVar') + assignin -> khong I/O, khong race.
%
% LUU Y: chay init_rwip_v13 va patch_rwip_model TRUOC file nay.
% ===========================================================================

clc;
modelName = 'RWIP_PSO_FLC';

%% ---------------- USER SETTINGS ----------------
StopTime  = '30';
thetaRef  = 0;
phiRef    = 0;

useParallel = false;   % dat true neu da co Parallel Computing Toolbox

% --- cua so danh gia ---
Tos     = 4.0;    % cua so xet vot lo
TendWin = 8.0;    % cua so xet xac lap (dai hon v12 vi vong ngoai phi CHAM)
% DUNG SAI XAC LAP -- PHAI GIONG HET sim_rwip_v13.m, neu khong bang so lieu
% cuoi cung se khong nhat quan voi ham muc tieu da toi uu.
% (Da tung lech: pso dung 0.02/0.10 trong khi sim dung 0.002/0.020. PSO khi do
%  toi uu ts_phi tren mot moc long gap 5 lan va tra ve nghiem co ts_phi=16.1 s
%  do bang thang chat -- KEM hon ca particle gieo ban dau 10.5 s.)
thetaTol = 0.002;  % rad  = 2% cua theta0 = 0.10 (tieu chuan 2% kinh dien)
phiTol   = 0.020;  % rad

thetaHardLimit     = 1.2;    % rad -- qua nguong nay coi nhu mat kha nang phuc hoi
thetaExceedPenalty = 1e5;
% J_simError PHAI lon hon moi J HOP LE, neu khong PSO se UU TIEN particle lam
% sim crash hon particle chay duoc nhung phan ky.
% (Do duoc: particle s=0.9 phan ky cho gv=2411 -> J = wCon2*gv^2 = 5.8e10,
%  trong khi J_simError cu chi 1e8 -> crash duoc xep HANG TOT HON phan ky.)
J_simError         = 1e14;

% --- trong so cost ---
% Da can chinh lai theo THANG DO THUC (phi khong con bi wrap nen IAE_phi gio
% co the len toi hang tram; neu giu q_phi=8 nhu v12 thi so hang nay se ap dao
% moi thu khac). Ty trong muc tieu tai nghiem toi uu: theta ~35%, phi ~35%,
% nang luong/chattering ~20%, phat xac lap ~10%.
% wOS_theta: sau khi OS_theta do VOT LO THUC (xem eval), gia tri dien hinh giam
% tu ~0.1 xuong ~0.012 -> binh phuong giam 70 lan. Tang trong so tuong ung de
% so hang nay co ty trong that (0.012^2 * 20000 = 2.9, dang ke so voi J ~ 11).
% ===========================================================================
% DAC TA HIEU NANG -- DAT TRUOC KHI TOI UU, KHONG SUA SAU KHI XEM KET QUA.
% ---------------------------------------------------------------------------
% Day la RANG BUOC, khong phai trong so. PSO tim nghiem TIET KIEM NHAT trong
% so cac nghiem DAT du 4 dieu kien. Cac muc duoi day nam sat bien kha thi cua
% kien truc (do bang mo phong doc lap: nghiem tot nhat dat OS=0.0058,
% ts_theta=2.67 s, phi_max=0.495 rad, ts_phi=10.5 s).
% Neu siet chat hon, bai toan VO NGHIEM: PSO se tra ve nghiem vi pham it nhat
% va ban phai NOI LONG dac ta rooi noi ro trong bai bao -- KHONG duoc lang le
% doi trong so cho tra ve ket qua mong muon.
% HIEU CHINH theo dung sai CHAT (thetaTol=0.002, phiTol=0.020). Cac muc duoi
% day lay tu diem KHA THI DA DO DUOC (nghiem gieo particle 1):
%       OS=0.0057  ts_theta=2.67 s  phi_max=0.493  ts_phi=10.47 s
% roi noi ra ~5% de bai toan co mien kha thi khac rong (neu dat dung bang diem
% da biet thi PSO chi con dung mot nghiem duy nhat va khong the cai thien).
spec_OS      = 0.006;   % rad  vot lo theta qua diem dat
spec_tsTheta = 3.0;     % s    |theta| < thetaTol tu thoi diem nay tro di
spec_phiMax  = 0.52;    % rad  dinh cua |phi|
spec_tsPhi   = 11.0;    % s    |phi| < phiTol tu thoi diem nay tro di
wCon  = 1e3;            % phat vi pham, tuyen tinh
wCon2 = 1e4;            % phat vi pham, bac hai (gradient manh ve mien kha thi)

% --- Trong so cua HAM MUC TIEU (nang luong + do muot + giu nghiem "sach") ---
% CANH BAO: q_theta, q_phi, wOS_theta, wOS_phi, wSettle KHONG con nam trong J
% sau khi doi sang dang co rang buoc. Giu lai CHI de tinh va bao cao IAE/OS
% trong struct M. Doi chung KHONG anh huong gi toi ket qua toi uu -- neu muon
% doi trong so hieu nang, phai doi DAC TA o tren.
q_theta   = 20;     wEnd_theta = 400;   wOS_theta = 20000;   % (chi bao cao)
q_phi     = 0.5;    wEnd_phi   = 60;    wOS_phi   = 1.0;     % (chi bao cao)
wSlope_phi   = 400;              % do doc phi cuoi
wPhiDotEnd   = 200;              % |phi_dot| cuoi: bat buoc banh da phai DUNG
ru = 0.002;  rdu = 2e-4;  rtvdu = 1e-4;
wSettle = 60;
rGain   = 1e-4;

% --- PSO ---
MaxIt = 20;  nPop = 30;
w = 0.9;  wdamp = 0.97;  c1 = 1.5;  c2 = 1.7;

%% ---------------- KHONG GIAN TIM KIEM ----------------
varNames = {'s_e','s_de','s_u','Ke','Kde','Ku','Kd_theta','Kd_phi','Kp_phi','Kv_phi'};
VarMin   = [0.15 0.15 0.15 0.5  0.05  4.0  0.0 -4.0  0.000  0.000];
VarMax   = [0.95 0.95 0.95 6.0  1.00 24.0  4.0  0.4  0.030  0.120];
nVar     = numel(VarMin);

% --- Rang buoc bien 8 (Kd_phi): CAN TREN 0.4 < Ke_eff = 0.5 ---
% tau = (Kt/Ra)*(V - Ke*phi_dot); nhanh Kd_phi cho tau ~ (Kd_phi - Ke)*phi_dot.
% Moi Kd_phi > 0.5 la PHAN HOI DUONG tren banh da -> chay den bao hoa +-24 V.
% Gia tri v12 (3.57697) nam sau trong vung nay.
%
% --- Rang buoc bien 9-10 (|Kp_phi|,|Kv_phi|); dau do signOuter quyet dinh ---
% G = mgl_eq/Ir = 29.6 1/s^2 (suy truc tiep tu phuong trinh, khong uoc luong).
%   G*|Kp| <= 0.89 -> cuc cham <= ~0.94 rad/s
%   G*|Kv| <= 3.55 -> cuc nhanh <= ~3.55 rad/s
% => ca hai cuc vong ngoai luon duoi w_pend = 5.21 va duoi wn vong trong 12.3.
% Noi long cac bien nay se cho PSO tim nghiem "nhanh" nhung pha vo tach thang
% thoi gian roi dao dong -- dung noi long.
signOuter = -1;   % theta_ref = -(Kp*phi + Kv*phi_dot); xac nhan lai bang signcheck

% VelMax duoc tinh SAU khi co nghiem gieo (xem phan KHOI TAO QUAN THE), vi no
% phai ty le voi GIA TRI tham so chu khong voi do rong mien.

%% ---------------- CHUAN BI MODEL ----------------
if ~bdIsLoaded(modelName), load_system(modelName); end

fuzzyBlk = find_fuzzy_block(modelName);   % ben hon find_system('BlockType','Reference')

% (C) Trong luc PSO chay, khoi Fuzzy doc FIS tu BIEN workspace 'fisVar' chu
% khong tu file. Nho lai gia tri cu va TU KHOI PHUC khi ham ket thuc (ke ca khi
% loi hoac Ctrl-C) -- neu khong, sau khi chay xong khoi se ket lai o 'fisVar'
% va moi lan sim() sau do se loi "Undefined variable fisVar" khi workspace da
% bi clear.
fisParamOld = get_param(fuzzyBlk,'FIS');
restoreFIS  = onCleanup(@() restoreFisParam(fuzzyBlk, fisParamOld));
fprintf('Khoi Fuzzy: %s  -> ''fisVar'' (se khoi phuc khi ket thuc)\n', fisParamOld);
set_param(fuzzyBlk,'FIS','fisVar');

% Dat StopTime/SaveFormat MOT LAN tren model. Cac tham so nay khong tunable
% duoi fast restart nen phai tat fast restart truoc khi dat, roi bat lai.
fr0 = get_param(modelName,'FastRestart');
if strcmpi(fr0,'on'), set_param(modelName,'FastRestart','off'); end
set_param(modelName,'StopTime',  StopTime);
try, set_param(modelName,'SaveFormat','Dataset'); catch, end   % ten tham so tuy phien ban
ensureLoggerTimeseries(modelName, {'theta','phi','u','phi_dot'});
if strcmpi(fr0,'on'), set_param(modelName,'FastRestart','on'); end

% Hang so vat ly / dieu kien dau -- lay tu base workspace (do init_rwip_v13 tao)
reqVars = {'p','theta0','theta_dot0','phi0','phi_dot0','theta_ref_max'};
modelConsts = struct();
for k = 1:numel(reqVars)
    vn = reqVars{k};
    assert(evalin('base', sprintf('exist(''%s'',''var'')==1', vn)) == 1, ...
        'Thieu bien "%s" trong base workspace. Hay chay init_rwip_v13 truoc.', vn);
    modelConsts.(vn) = evalin('base', vn);
end

ctx = struct();
ctx.modelName = modelName;   ctx.StopTime = StopTime;
ctx.modelConsts = modelConsts;
ctx.thetaRef = thetaRef;     ctx.phiRef = phiRef;
ctx.Tos = Tos;               ctx.TendWin = TendWin;
ctx.thetaTol = thetaTol;     ctx.phiTol = phiTol;
ctx.thetaHardLimit = thetaHardLimit;
ctx.thetaExceedPenalty = thetaExceedPenalty;
ctx.J_simError = J_simError;
ctx.q_theta = q_theta;       ctx.q_phi = q_phi;
ctx.wOS_theta = wOS_theta;   ctx.wOS_phi = wOS_phi;
ctx.wEnd_theta = wEnd_theta; ctx.wEnd_phi = wEnd_phi;
ctx.wSlope_phi = wSlope_phi; ctx.wPhiDotEnd = wPhiDotEnd;
ctx.ru = ru; ctx.rdu = rdu; ctx.rtvdu = rtvdu;
ctx.wSettle = wSettle;       ctx.rGain = rGain;
ctx.spec_OS = spec_OS;             ctx.spec_tsTheta = spec_tsTheta;
ctx.spec_phiMax = spec_phiMax;     ctx.spec_tsPhi   = spec_tsPhi;
ctx.wCon = wCon;                   ctx.wCon2 = wCon2;
ctx.nPop = nPop;                   ctx.MaxIt = MaxIt;   % dung cho precheck
ctx.signOuter = signOuter;
ctx.debug = false;

if useParallel && isempty(gcp('nocreate'))
    try, parpool; catch, useParallel = false; end
end

%% ---------------- KHOI TAO QUAN THE ----------------
rng(20260806,'twister');

% ===========================================================================
% KHOI TAO CO DINH HUONG (informed initialization)
% ---------------------------------------------------------------------------
% Khoi tao NGAU NHIEN DEU tren toan mien KHONG dung duoc voi bai toan nay.
% Do duoc thuc te: trong 30 particle ngau nhien, 29 particle cho J >= 1.4e6
% trong khi nghiem thiet ke cho J = 87.3 -- kem hon 5 BAC DO LON. Hau qua:
% PSO chay voi 1 particle huu ich va 29 particle lang thang trong vung mat on
% dinh; gbest dung yen suot 10 vong lien tuc, khong nhuc nhich mot lan.
% Nguyen nhan: mien tham so on dinh chiem mot phan cuc nho cua khong gian 10
% chieu, khoi tao deu gan nhu khong bao gio roi vao do.
%
% Cach sua: phan lon quan the la NHIEU LOAN GAUSS quanh nghiem thiet ke giai
% tich, giu mot phan nho ngau nhien deu de van con kha nang thoat cuc bo.
%
% LUU Y KHI VIET BAI: voi khoi tao nay, PSO dong vai tro TINH CHINH CUC BO
% quanh mot thiet ke giai tich, KHONG phai tim kiem toan cuc. Phai mo ta dung
% nhu vay -- day la lua chon thiet ke co ly do, khong phai thieu sot.
% NHIEU LOAN THEO GIA TRI, KHONG theo do rong mien.
% (Da sai: dung sigma = 0.12*(VarMax-VarMin). Voi Kp_phi mien [0, 0.03] va
%  nghiem toi uu 0.0043, sigma do bang 0.0036 = 84% GIA TRI -> pha huy nghiem.
%  Ket qua do duoc: particle nhieu loan tot nhat J=8439, van kem nghiem gieo
%  87.3 toi 100 lan. Mien kha thi hep hon nhieu so voi do rong mien tim kiem.)
fracLocal  = 0.75;   % ty le particle nhieu loan quanh nghiem gieo
relSpread  = 0.08;   % do lech chuan = relSpread * |gia tri cua nghiem gieo|

Positions = VarMin + rand(nPop,nVar).*(VarMax - VarMin);   % nen ngau nhien deu
Velocities = zeros(nPop,nVar);

% Gieo 3 particle tu cac diem xuat phat co ly do
% Particle 1 = nghiem toi uu tim duoc bang PSO tren mo hinh vong kin dung lai
% doc lap (Python/RK4, da kiem chung khop Simulink: phi dinh 0.707 vs 0.708,
% theta cuc tieu -0.027 vs -0.027). J giam 6.67 -> 2.33.
% Nghiem tot nhat cua BAI TOAN CO RANG BUOC tren mo hinh Python (dat gan het
% dac ta: OS=0.0058 ts_th=2.67 phi_max=0.495 ts_phi=10.5).
Positions(1,:) = [0.2000 0.77182 0.31824 1.8823 0.66601 24.0 2.6788 0.19785 0.0043169 0.0184026];
% Particle 2 = thiet ke tay ban dau (dat cuc theo tach thang thoi gian)
Positions(2,:) = [0.5000 0.50000 0.5000 2.0000 0.33330 24.0 3.0000 0.00 0.011000 0.047000];
% Particle 3 = diem tot nhat lan chay truoc (khoi tao deu, vong 11): J=59.30,
% giam vi pham dac ta tu gv=0.0558 xuong ~0.027. Chi biet 7 gain tu log; ba
% tham so do rong MF giu theo nghiem gieo (log khong in ra).
Positions(3,:) = [0.2000 0.77182 0.31824 2.0030 0.69100 23.10 2.6060 0.15800 0.0044000 0.0188000];

% Particle 4..nLocal = nhieu loan Gauss quanh nghiem gieo (particle 1)
nLocal = max(4, round(fracLocal*nPop));
base   = Positions(1,:);
% sigma ty le voi GIA TRI. Voi tham so co gia tri 0 thi lay 2% do rong mien
% lam san de van co nhieu loan.
sig    = max(relSpread*abs(base), 0.02*(VarMax - VarMin).*(abs(base)<eps));
for i = 4:nLocal
    Positions(i,:) = min(max(base + sig.*randn(1,nVar), VarMin), VarMax);
end
% ===========================================================================
% GIOI HAN VAN TOC -- CUNG LY DO NHU sigma: theo GIA TRI, khong theo mien.
% ---------------------------------------------------------------------------
% Ban cu: VelMax = 0.2*(VarMax-VarMin). Do duoc tren nghiem toi uu:
%     Kd_phi : mien 4.40 -> VelMax 0.880 = 445% gia tri (0.198)
%     Kp_phi : mien 0.03 -> VelMax 0.006 = 140% gia tri (0.0043)
%     Kv_phi : mien 0.12 -> VelMax 0.024 = 130% gia tri (0.0184)
% Tuc MOT buoc van toc du de nem particle ra khoi mien kha thi ngay ca khi no
% dang dung dung nghiem tot. Day la ly do PSO khong tinh chinh duoc: sua khoi
% tao ma khong sua cho nay thi particle van bi van ra tu vong dau tien.
VelMax = max(0.25*abs(base), 0.01*(VarMax - VarMin));
VelMin = -VelMax;

fprintf('Khoi tao: %d particle quanh nghiem gieo (sigma=%.0f%% GIA TRI), %d ngau nhien deu.\n', ...
        nLocal, 100*relSpread, nPop-nLocal);
pctV = 100*VelMax./max(abs(base),eps);
fprintf('VelMax theo %% gia tri nghiem gieo:');
for k = 1:nVar, fprintf(' %s=%.0f%%', varNames{k}, pctV(k)); end
fprintf('\n');

if strcmpi(mode,'precheck')
    do_precheck(Positions, ctx, varNames, VarMin, VarMax);
    return
end

% --- CHOT CHAN: kiem tra buildFIS + 1 lan sim TRUOC khi chay ca quan the ---
fprintf('Kiem tra so bo...\n');
try
    fisTest = buildFIS(0.5, 0.5, 0.5);
    u00 = evalfis(fisTest, [0 0]);
    fprintf('  buildFIS OK, FLC(0,0) = %.3e %s\n', u00, ternary(abs(u00)<1e-9,'[OK]','[SAI]'));
catch ME
    error('buildFIS hong: %s\n(sua loi nay truoc, dung chay PSO)', ME.message);
end
% Trong TOAN BO phan kiem tra so bo, LUON in loi day du. (Da gap: bo dem loi
% nErr con luu tu lan chay truoc nen loi that bi giau, chi thay J=1e8.)
ctxDbg = ctx;  ctxDbg.debug = true;

[Jtest, Mtest] = evalCost(Positions(1,:), ctxDbg);
fprintf('  sim thu particle 1 : J = %.6g\n', Jtest);
if ~Mtest.simOK
    error(['sim() dang HONG (khong chay duoc), khong phai tham so xau.\n' ...
           'Xem thong bao [eval loi] ngay tren. Sua loi do truoc, dung chay PSO.']);
end

% --- CHOT CHAN 2: FIS co THUC SU thay doi giua cac lan danh gia khong? ---
% Duoi fast restart, khoi Fuzzy CO THE nap cung FIS o lan bien dich dau va
% KHONG doc lai bien 'fisVar'. Khi do MOI particle deu chay cung mot FIS ma
% khong he bao loi -- PSO van "hoi tu dep" tren mot bai toan gia.
% Kiem bang cach doi manh do rong MF roi xem J co doi khong.
xA = Positions(1,:);  xB = xA;  xB(1:3) = [0.90 0.90 0.90];
[JA, MA] = evalCost(xA, ctxDbg);  [JB, MB] = evalCost(xB, ctxDbg);
% Phan biet "sim hong" voi "FIS khong tac dong" bang CO simOK, khong bang nguong J.
if ~MA.simOK || ~MB.simOK
    error(['sim() HONG trong phep thu tac dong cua FIS (simOK: A=%d B=%d).\n' ...
           'Day KHONG phai loi fast restart. Xem [eval loi] o tren.'], MA.simOK, MB.simOK);
end
if abs(JA-JB) <= 1e-9*max(1,abs(JA))
    if strcmpi(get_param(modelName,'FastRestart'),'on')
        set_param(modelName,'FastRestart','off');
    end
    error(['FIS KHONG tac dong: doi do rong MF tu (%.2f,%.2f,%.2f) sang ' ...
           '(0.9,0.9,0.9) cho J y het nhau (%.10g).\n' ...
           'Khoi Fuzzy dang khong doc lai bien fisVar -- rat co the do fast ' ...
           'restart nap cung FIS. Da tu tat FastRestart; hay chay lai.\n' ...
           'NEU van loi khi FastRestart=off thi van de nam o cho khac, dung ' ...
           'chay PSO cho toi khi giai quyet xong.'], xA(1),xA(2),xA(3), JA);
end
fprintf('  FIS co tac dung  : J(s=%.2f)=%.6g vs J(s=0.9)=%.6g  [OK]\n', xA(1), JA, JB);
% (Kiem tra J >= J_simError da bo: da thay bang co Mtest.simOK o tren, tin cay
%  hon vi J HOP LE co the vuot moi nguong khi vi pham rang buoc nang.)

Costs = inf(nPop,1);
fprintf('Danh gia quan the ban dau (%d particle)...\n', nPop);
if useParallel
    parfor i = 1:nPop, Costs(i) = evalCost(Positions(i,:), ctx); end
else
    for i = 1:nPop
        Costs(i) = evalCost(Positions(i,:), ctx);
        fprintf('  particle %2d/%2d : J = %.6g\n', i, nPop, Costs(i));
    end
end

PBestPos = Positions;  PBestCost = Costs;
[GBestCost, gi] = min(Costs);  GBestPos = Positions(gi,:);
BestHist = zeros(MaxIt,1);

%% ---------------- VONG LAP PSO ----------------
for it = 1:MaxIt
    for i = 1:nPop
        r1 = rand(1,nVar); r2 = rand(1,nVar);
        Velocities(i,:) = w*Velocities(i,:) ...
            + c1*r1.*(PBestPos(i,:) - Positions(i,:)) ...
            + c2*r2.*(GBestPos      - Positions(i,:));
        Velocities(i,:) = min(max(Velocities(i,:), VelMin), VelMax);
        Positions(i,:)  = Positions(i,:) + Velocities(i,:);
        % phan xa tai bien (giu da dang tot hon la kep cung)
        lo = Positions(i,:) < VarMin;  hi = Positions(i,:) > VarMax;
        Positions(i,lo) = VarMin(lo);  Positions(i,hi) = VarMax(hi);
        Velocities(i, lo | hi) = -0.5*Velocities(i, lo | hi);
    end

    if useParallel
        parfor i = 1:nPop, Costs(i) = evalCost(Positions(i,:), ctx); end
    else
        for i = 1:nPop, Costs(i) = evalCost(Positions(i,:), ctx); end
    end

    imp = Costs < PBestCost;
    PBestPos(imp,:) = Positions(imp,:);  PBestCost(imp) = Costs(imp);
    [mc, mi] = min(PBestCost);
    if mc < GBestCost, GBestCost = mc; GBestPos = PBestPos(mi,:); end

    BestHist(it) = GBestCost;
    w = w*wdamp;
    % In DAY DU 10 tham so. (Da mat mot nghiem tot vi log cu bo qua s_e/s_de/s_u
    % nen khong tai lap duoc diem do tu log.)
    fprintf('Iter %2d/%d | J = %-12.6g | s=[%.4f %.4f %.4f] Ke=%.4f Kde=%.4f Ku=%.3f Kd_th=%.4f Kd_phi=%.4f Kp=%.5f Kv=%.5f\n', ...
        it, MaxIt, GBestCost, GBestPos(1), GBestPos(2), GBestPos(3), GBestPos(4), ...
        GBestPos(5), GBestPos(6), GBestPos(7), GBestPos(8), GBestPos(9), GBestPos(10));
    % Luu tien do sau MOI vong -- mat dien/Ctrl+C khong lam mat ket qua.
    try, save('pso_rwip_v13_progress.mat','GBestPos','GBestCost','BestHist','it','varNames'); catch, end
end

%% ---------------- LUU KET QUA ----------------
best = struct();
for k = 1:nVar, best.(varNames{k}) = GBestPos(k); end
best.Kp_phi = ctx.signOuter * best.Kp_phi;
best.Kv_phi = ctx.signOuter * best.Kv_phi;

fisBest = buildFIS(GBestPos(1), GBestPos(2), GBestPos(3));
saveFISFile(fisBest, 'fis_rwip_v13_best.fis');

% Danh gia lai nghiem tot nhat de LUU KEM chi so vat ly + muc vi pham dac ta.
% (Khong the tai lap tu J: J la tong cua muc tieu va phat, khong tach nguoc duoc.)
[~, Mbest] = evalCost(GBestPos, ctx);
spec = struct('OS',ctx.spec_OS,'tsTheta',ctx.spec_tsTheta, ...
              'phiMax',ctx.spec_phiMax,'tsPhi',ctx.spec_tsPhi, ...
              'thetaTol',ctx.thetaTol,'phiTol',ctx.phiTol);
save('pso_rwip_v13_result.mat','best','GBestPos','GBestCost','BestHist', ...
     'varNames','Mbest','spec','VarMin','VarMax','VelMax','relSpread','fracLocal');

fprintf('\n--- Doi chieu voi DAC TA ---\n');
chkSpec('vot lo theta', Mbest.OS_theta, ctx.spec_OS);
chkSpec('ts_theta',     Mbest.ts_theta, ctx.spec_tsTheta);
chkSpec('phi_max',      Mbest.phi_peak, ctx.spec_phiMax);
chkSpec('ts_phi',       Mbest.ts_phi,   ctx.spec_tsPhi);
if Mbest.gv > 0
    fprintf(2,['\n(!) Con vi pham dac ta %.2f%%. Theo nguyen tac da dat: BAO CAO\n' ...
               '    dung nhu vay, NOI dac ta va ghi ro trong bai bao neu can --\n' ...
               '    KHONG duoc doi trong so cho ra ket qua mong muon.\n'], 100*Mbest.gv);
else
    fprintf('\nDAT DU DAC TA (gv = 0).\n');
end

fprintf('\n===================== KET QUA =====================\n');
fprintf(' J_best = %.6g\n', GBestCost);
for k = 1:nVar
    fprintf('   %-10s = %.6g\n', varNames{k}, best.(varNames{k}));
end
fprintf('\n Da luu: fis_rwip_v13_best.fis , pso_rwip_v13_result.mat\n');
fprintf(' De ap dung: cap nhat cac gia tri tren vao init_rwip_v13.m,\n');
fprintf('             va doi FIS cua khoi Fuzzy sang ''fis_rwip_v13_best.fis''.\n');
fprintf('==================================================\n');

figure('Name','PSO v13 convergence');
plot(1:MaxIt, BestHist, 'LineWidth', 1.4); grid on;
xlabel('Iteration'); ylabel('Best cost J'); title('PSO v13 - hoi tu');
end


%% =========================================================================
function [J, M] = evalCost(x, ctx)
% simOK: co TUONG MINH bao sim chay duoc hay khong.
% KHONG duoc nhan biet loi bang cach so J voi mot nguong: J HOP LE co the vuot
% moi nguong (particle phan ky cho gv=2411 -> J = 1e4*gv^2 = 5.8e10).
M = struct('simOK',false,'OS_theta',NaN,'ts_theta',NaN,'phi_peak',NaN,'ts_phi',NaN, ...
           'IAE_theta',NaN,'IAE_phi',NaN,'Eu',NaN,'Edu',NaN,'tv_du',NaN, ...
           'theta_end',NaN,'phi_end',NaN,'phid_end',NaN,'gv',NaN,'Jobj',NaN);
try
    fis = buildFIS(x(1), x(2), x(3));

    assignin('base','fisVar',  fis);
    assignin('base','Ke',      x(4));
    assignin('base','Kde',     x(5));
    assignin('base','Ku',      x(6));
    assignin('base','Kd_theta',x(7));
    assignin('base','Kd_phi',  x(8));
    assignin('base','Kp_phi',  ctx.signOuter * x(9));
    assignin('base','Kv_phi',  ctx.signOuter * x(10));
    fn = fieldnames(ctx.modelConsts);
    for k = 1:numel(fn), assignin('base', fn{k}, ctx.modelConsts.(fn{k})); end

    % Fast restart KHONG cho doi StopTime/SaveFormat luc chay -> khi bat fast
    % restart phai dua chung qua set_param truoc (xem phan CHUAN BI MODEL).
    % Khi KHONG bat fast restart, giu nguyen cach goi cu (da chay on dinh).
    if strcmpi(get_param(ctx.modelName,'FastRestart'),'on')
        simOut = sim(ctx.modelName);
    else
        simOut = sim(ctx.modelName,'StopTime',ctx.StopTime,'SaveFormat','Dataset');
    end

    [t, theta] = ts2tv(simOut,'theta');
    [~, phi]   = ts2tv(simOut,'phi');
    [~, u]     = ts2tv(simOut,'u');
    [~, phid]  = ts2tv(simOut,'phi_dot');

    if isempty(t) || isempty(theta) || isempty(phi) || isempty(u)
        J = ctx.J_simError; return
    end
    N = min([numel(t) numel(theta) numel(phi) numel(u)]);
    t = t(1:N); theta = theta(1:N); phi = phi(1:N); u = u(1:N);
    if numel(phid) >= N, phid = phid(1:N); else, phid = [0; diff(phi)./diff(t)]; end

    dt = median(diff(t)); if ~isfinite(dt) || dt <= 0, dt = 0.01; end

    % ===== (A) KHONG WRAP PHI =====
    % phi la goc BANH DA -- mot vat quay lien tuc. Wrap no ve [-pi,pi] xoa sach
    % thong tin ve tich luy vong quay va bien moi so hang phat phi thanh nhieu.
    e_theta = theta - ctx.thetaRef;
    e_phi   = phi   - ctx.phiRef;         % <-- nguyen ban, khong mod()

    IAE_theta = trapz(t, abs(e_theta));
    IAE_phi   = trapz(t, abs(e_phi));

    idxOS = t <= ctx.Tos;  if ~any(idxOS), idxOS = true(size(t)); end
    % VOT LO THUC = phan di QUA diem dat, o phia DOI DIEN voi diem xuat phat.
    % (Ban cu: max|theta| - |thetaRef|. Voi thetaRef=0 no bang max|theta|, ma
    %  gia tri do luon ~ theta0 = 0.1 voi MOI nghiem on dinh -> la HANG SO,
    %  khong phan biet duoc particle nao tot hon. Tuc PSO hoan toan KHONG
    %  "nhin thay" vot lo, dung thu ma ta muon toi thieu hoa.)
    if theta(1) >= ctx.thetaRef
        OS_theta = max(0, ctx.thetaRef - min(theta(idxOS)));   % vot xuong duoi
    else
        OS_theta = max(0, max(theta(idxOS)) - ctx.thetaRef);   % vot len tren
    end
    OS_phi   = max(abs(e_phi(idxOS)));

    Eu    = trapz(t, u.^2);
    du    = [0; diff(u)]/dt;
    Edu   = trapz(t, du.^2);
    tv_du = sum(abs(diff(du)));

    idxEnd = t >= max(t(1), t(end) - ctx.TendWin);
    if ~any(idxEnd), idxEnd = true(size(t)); end

    theta_end  = theta(end);
    phi_end    = phi(end);
    phid_end   = mean(abs(phid(idxEnd)));   % banh da phai THUC SU dung lai

    tt = t(idxEnd); pp = phi(idxEnd);
    if numel(tt) >= 5
        A = [tt(:) ones(numel(tt),1)];
        cf = A\pp(:);  phi_slope = cf(1);
    else
        phi_slope = (phi(end) - phi(max(1,end-1)))/dt;
    end

    settle_pen = ( sum(abs(theta(idxEnd)-ctx.thetaRef) > ctx.thetaTol) ...
                 + sum(abs(phi(idxEnd)  -ctx.phiRef)   > ctx.phiTol) ) * dt;

    % ===================================================================
    % DANG CO RANG BUOC (thay cho tong-co-trong-so)
    % -------------------------------------------------------------------
    % LY DO DOI: voi tong-co-trong-so, nghiem phu thuoc hoan toan vao bo
    % trong so ta chon, va co the "chinh" trong so de bat ky ket luan nao
    % cung dung. Da quan sat truc tiep: bo trong so ban dau cho ra nghiem
    % ts_theta=8.5 s / phi_max=0.87 (muot nhung cham); tang q_phi/wSettle
    % lai cho vot lo 0.042 (nhanh nhung vot). Khong bo nao "dung" hon bo nao.
    %
    % Dang co rang buoc tach bach hai thu:
    %   MUC TIEU   : nang luong dieu khien + do muot  (toi thieu hoa)
    %   RANG BUOC  : dac ta hieu nang, DAT TRUOC khi toi uu (khong duoc sua
    %                sau khi thay ket qua -- do la diem mau chot ve tinh
    %                giai trinh duoc trong bai bao)
    % Nghiem thu duoc la "re nhat trong so cac nghiem DAT dac ta", mot phat
    % bieu ro rang va kiem chung duoc, thay vi "toi uu theo mot bo trong so
    % khong ai giai thich duoc tai sao lai la bo do".
    % ===================================================================
    ts_theta = lastTimeAbove(t, theta - ctx.thetaRef, ctx.thetaTol);
    ts_phi   = lastTimeAbove(t, phi   - ctx.phiRef,   ctx.phiTol);
    phi_peak = max(abs(e_phi));

    % Vi pham tuong doi cua tung rang buoc (0 = dat)
    gv = max(0, OS_theta/ctx.spec_OS      - 1) ...
       + max(0, ts_theta/ctx.spec_tsTheta - 1) ...
       + max(0, phi_peak/ctx.spec_phiMax  - 1) ...
       + max(0, ts_phi  /ctx.spec_tsPhi   - 1);

    % Muc tieu: nang luong + do muot (+ cac so hang giu cho nghiem "sach")
    Jobj = ctx.ru*Eu + ctx.rdu*Edu + ctx.rtvdu*tv_du ...
         + ctx.wEnd_theta*((theta_end-ctx.thetaRef)^2) ...
         + ctx.wEnd_phi*((phi_end-ctx.phiRef)^2) ...
         + ctx.wSlope_phi*(phi_slope^2) ...
         + ctx.wPhiDotEnd*(phid_end^2) ...
         + ctx.rGain*(x(7)^2 + x(8)^2);

    % Phat vi pham: tuyen tinh + bac hai -> gradient manh va lien tuc, dan
    % particle ve mien kha thi thay vi chi loai bo cung.
    J = Jobj + ctx.wCon*gv + ctx.wCon2*(gv^2);

    % Xuat toan bo chi so de precheck kiem tra duoc (chi so "chet", lech dinh nghia)
    M.OS_theta=OS_theta; M.ts_theta=ts_theta; M.phi_peak=phi_peak; M.ts_phi=ts_phi;
    M.IAE_theta=IAE_theta; M.IAE_phi=IAE_phi; M.Eu=Eu; M.Edu=Edu; M.tv_du=tv_du;
    M.theta_end=theta_end; M.phi_end=phi_end; M.phid_end=phid_end;
    M.gv=gv; M.Jobj=Jobj; M.simOK=true;

    % Giu lai IAE de bao cao (khong vao muc tieu nua)
    if ctx.debug
        fprintf(2,'[dbg] Jobj=%.4g gv=%.4g | OS=%.4g ts_th=%.3g phi_pk=%.3g ts_phi=%.3g | IAEt=%.3g IAEp=%.3g\n', ...
            Jobj, gv, OS_theta, ts_theta, phi_peak, ts_phi, IAE_theta, IAE_phi);
    end

    % Phat cung neu con lac nga qua nguong khong the phuc hoi, o BAT KY luc nao
    if max(abs(theta)) > ctx.thetaHardLimit
        J = J + ctx.thetaExceedPenalty;
    end

    if ~isfinite(J), J = ctx.J_simError; end

    if ctx.debug
        fprintf(2,['[dbg] J=%.6g | IAEth=%.4g IAEphi=%.4g OSth=%.4g ' ...
                   'phi_end=%.4g slope=%.4g phid_end=%.4g\n'], ...
            J, IAE_theta, IAE_phi, OS_theta, phi_end, phi_slope, phid_end);
    end
catch ME
    % QUAN TRONG: LUON bao 3 loi dau tien. Truoc day catch nuot sach moi loi
    % thanh J=1e8, nen mot bug trong buildFIS lam CA quan the = 1e8 ma van chay
    % het 20 vong roi moi lo ra o dong cuoi. Khong bao gio de loi im lang nua.
    persistent nErr
    if isempty(nErr), nErr = 0; end
    nErr = nErr + 1;
    if nErr <= 3 || ctx.debug
        fprintf(2,'\n[eval loi #%d] %s\n', nErr, ME.message);
        for k = 1:min(3,numel(ME.stack))
            fprintf(2,'    tai %s (dong %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
        if nErr == 3
            fprintf(2,'    (khong bao cac loi tiep theo; bat ctx.debug=true de xem het)\n');
        end
    end
    J = ctx.J_simError;
end
end


%% =========================================================================
function fis = buildFIS(s_e, s_de, s_u)
%BUILDFIS  Dung FIS Mamdani 2-vao 1-ra, universe [-1,1], DOI XUNG hoan toan.
%
% (B) Ba tham so s_* la nua-do-rong cua MF trung tam. Vi MF luon doi xung qua 0
% va bang luat phan doi xung, FLC(0,0) = 0 duoc BAO DAM VE MAT CAU TRUC --
% khong the tai dien loi bias +4.6 V cua fis_rwip_best_2.fis.
%
% Bang luat: u(i,j) = clamp(9-i-j, 1, 5), i=e(1..5=NB..PB), j=de(1..5).
% Day la bang PD mo kinh dien, thoa u(i,j) + u(6-i,6-j) = 6 voi moi i,j.

s_e  = min(max(s_e , 0.10), 0.95);
s_de = min(max(s_de, 0.10), 0.95);
s_u  = min(max(s_u , 0.10), 0.95);

fis = mamfis('Name','fis_rwip_sym', ...
    'AndMethod','min','OrMethod','max','ImplicationMethod','min', ...
    'AggregationMethod','max','DefuzzificationMethod','centroid');

fis = addInput (fis, [-1 1], 'Name','e_theta');
fis = addInput (fis, [-1 1], 'Name','de_theta');
fis = addOutput(fis, [-1 1], 'Name','u');

% addMF tra bien theo TEN, khong phai theo dinh danh 'input1'/'output1'.
fis = addSymMFs(fis, 'e_theta',  s_e);
fis = addSymMFs(fis, 'de_theta', s_de);
fis = addSymMFs(fis, 'u',        s_u);

% BANG LUAT: u(i,j) = clamp(i+j-3, 1, 5)   -> e DUONG cho u DUONG.
% CUC TINH NAY LA BAT BUOC, suy tu RWIP_Dynamics:
%     theta_ddot = (-tau + ... + mgl_eq*sin(theta) - ...)/Ip_eq
%     tau        = (Kt/Ra)*(V - Ke*phi_dot)
% => d(theta_ddot)/dV = -(Kt/Ra)/Ip_eq = -2.79  (AM)
% Nen theta > 0 (dang do ve phia duong) can theta_ddot < 0, tuc can V > 0.
% Bang cu clamp(9-i-j,1,5) cho e>0 -> u<0 la NGUOC DAU va gay phan ky.
rl = zeros(25,4);  r = 0;
for i = 1:5
    for j = 1:5
        r = r + 1;
        rl(r,:) = [i, j, min(5, max(1, i+j-3)), 1];
    end
end
fis = addRule(fis, [rl ones(25,1)]);
end

function fis = addSymMFs(fis, varName, s)
nm = {'NB','NS','Z','PS','PB'};
P  = { [-1 -1 -1 -s], [-1 -s 0], [-s 0 s], [0 s 1], [s 1 1 1] };
T  = { 'trapmf','trimf','trimf','trimf','trapmf' };
for k = 1:5
    fis = addMF(fis, varName, T{k}, P{k}, 'Name', nm{k});
end
end

function saveFISFile(fis, fileName)
% Ghi nguyen tu (temp -> move) de khong bao gio de lai file .fis do dang.
tmp = [tempname '.fis'];
try
    writeFIS(fis, tmp);        % R2018b+
catch
    writefis(fis, tmp);        % ban cu
end
if exist(fileName,'file')==2
    copyfile(fileName, sprintf('%s.bak_%s', fileName, datestr(now,'yyyymmdd_HHMMSS')));
end
movefile(tmp, fileName, 'f');
end


%% =========================================================================
function ensureLoggerTimeseries(modelName, varNames)
for k = 1:numel(varNames)
    blk = find_system(modelName,'BlockType','ToWorkspace','VariableName',varNames{k});
    for b = 1:numel(blk)
        try
            set_param(blk{b},'SaveFormat','Timeseries');
            set_param(blk{b},'MaxDataPoints','inf');
        catch, end
    end
end
end

function restoreFisParam(blk, oldVal)
try
    set_param(blk, 'FIS', oldVal);
    fprintf('\n[cleanup] Khoi Fuzzy da tro lai: %s\n', oldVal);
catch
end
end

function do_precheck(Positions, ctx, varNames, VarMin, VarMax)
%DO_PRECHECK  Kiem tra ham muc tieu TRUOC khi bo ra 40-90 phut toi uu.
fprintf('\n============ PRECHECK ============\n');

% ---- 0. FIS ----
fis = buildFIS(0.5,0.5,0.5); u00 = evalfis(fis,[0 0]);
fprintf('[0] FLC(0,0) = %.3e  %s\n', u00, ternary(abs(u00)<1e-9,'OK','SAI'));

% ---- 1. Danh gia mot nhom nghiem DA DANG ----
nT = 6;
X = [Positions(1:3,:); ...
     VarMin + 0.25*(VarMax-VarMin); ...
     VarMin + 0.50*(VarMax-VarMin); ...
     VarMin + 0.75*(VarMax-VarMin)];
fprintf('[1] Danh gia %d nghiem da dang...\n', nT);
tic; J = zeros(nT,1); Ms = [];
for i = 1:nT
    [J(i), m] = evalCost(X(i,:), ctx);
    if isempty(Ms), Ms = m; else, Ms(i) = m; end %#ok<AGROW>
    fprintf('    #%d  J = %-14.6g gv = %-10.4g\n', i, J(i), m.gv);
end
tEval = toc/nT;
fprintf('    ~%.1f s/eval  ->  uoc tinh toan bo PSO: %.0f phut\n', ...
        tEval, tEval*(numel(J)*0+ctx.nPop*(ctx.MaxIt+1))/60);

% ---- 2. Chi so CHET ----
fprintf('\n[2] Chi so CHET (khong bien thien giua cac nghiem = PSO khong "nhin thay")\n');
fn = fieldnames(Ms);
ok2 = true;
for k = 1:numel(fn)
    v = [Ms.(fn{k})]; v = v(isfinite(v));
    if numel(v) < 3, continue; end
    cv = std(v)/max(abs(mean(v)), eps);
    if cv < 0.02
        fprintf(2,'    %-12s CV = %.4f  <-- CHET (gia tri ~ %.4g o moi nghiem)\n', fn{k}, cv, mean(v));
        ok2 = false;
    else
        fprintf('    %-12s CV = %.3f  OK\n', fn{k}, cv);
    end
end
if ok2, fprintf('    -> khong co chi so chet.\n'); end

% ---- 3. LECH DINH NGHIA so voi sim_rwip_v13 ----
fprintf('\n[3] Doi chieu dinh nghia chi so voi sim_rwip_v13 (nghiem gieo #1)\n');
try
    x = Positions(1,:);
    % QUAN TRONG: phai tai tao FIS tu x(1:3) chu KHONG dung file co san.
    % (Da gap: dung fis_rwip_opt.fis co s=(0.20,0.551,0.200) trong khi particle
    %  #1 co s=(0.20,0.772,0.318) -> lech ~19% dong deu o 4 chi so, va bi bao
    %  nham la "lech dinh nghia". Do la loi cua chinh precheck.)
    % Ten DUY NHAT moi lan goi: sim_rwip_v13 cache mat dieu khien theo TEN FILE,
    % dung ten co dinh se doc phai cache cu neu chay precheck lan thu hai.
    fTmp = [tempname '.fis'];
    try, writeFIS(buildFIS(x(1),x(2),x(3)), fTmp);
    catch, writefis(buildFIS(x(1),x(2),x(3)), fTmp); end
    P = make_P_v13(fTmp);
    P.Ke=x(4); P.Kde=x(5); P.Ku=x(6); P.Kd_theta=x(7); P.Kd_phi=x(8);
    P.Kp_phi_mag=x(9); P.Kv_phi_mag=x(10);
    o = sim_rwip_v13('PSOFLC', P, [ctx.modelConsts.theta0;0;0;0], ...
                     str2double(ctx.StopTime), 1e-3);
    pairs = {'OS_theta','OS_theta'; 'ts_theta','ts_theta'; 'ts_phi','ts_phi'; ...
             'phi_peak','phi_max'; 'IAE_theta','IAEtheta'; 'IAE_phi','IAEphi'};
    ok3 = true;
    for k = 1:size(pairs,1)
        a = Ms(1).(pairs{k,1}); b = o.(pairs{k,2});
        d = abs(a-b)/max(abs(b),eps)*100;
        s = ternary(d<10,'OK','<-- LECH');
        if d>=10, ok3=false; fprintf(2,'    %-11s cost=%-11.5g sim=%-11.5g  %5.1f%%  %s\n',pairs{k,1},a,b,d,s);
        else,     fprintf('    %-11s cost=%-11.5g sim=%-11.5g  %5.1f%%  %s\n',pairs{k,1},a,b,d,s); end
    end
    if ok3, fprintf('    -> dinh nghia nhat quan.\n'); end
catch ME
    fprintf(2,'    khong doi chieu duoc: %s\n', ME.message);
end

% ---- 4. MIEN KHA THI + moc so sanh ----
fprintf('\n[4] Dac ta va mien kha thi\n');
fprintf('    OS<=%.4g  ts_theta<=%.4g  phi_max<=%.4g  ts_phi<=%.4g\n', ...
        ctx.spec_OS, ctx.spec_tsTheta, ctx.spec_phiMax, ctx.spec_tsPhi);
m = Ms(1);
fprintf('    nghiem gieo #1: OS=%.5g  ts_theta=%.4g  phi_max=%.4g  ts_phi=%.4g  -> gv=%.4g\n', ...
        m.OS_theta, m.ts_theta, m.phi_peak, m.ts_phi, m.gv);
if m.gv > 0
    fprintf(2,'    (!) Nghiem gieo VI PHAM dac ta %.1f%%. PSO se phai tim nghiem tot hon no,\n', m.gv*100);
    fprintf(2,'        hoac dac ta la KHONG KHA THI. Neu sau khi chay PSO gv van > 0 thi\n');
    fprintf(2,'        NOI dac ta va ghi ro trong bai bao -- dung doi trong so.\n');
else
    fprintf('    nghiem gieo DAT dac ta. PSO se toi thieu hoa nang luong trong mien kha thi.\n');
end
fprintf('\n    MOC SO SANH: J(gieo) = %.6g\n', J(1));
fprintf('    PSO phai ha duoc duoi moc nay, neu khong -> bao cao chinh nghiem gieo.\n');

fprintf('\n============ HET PRECHECK ============\n');
fprintf('Neu tat ca OK:  pso_rwip_v13\n\n');
end

function chkSpec(name, val, lim)
if val <= lim
    fprintf('  %-14s = %9.5g   <= %-9.5g  DAT\n', name, val, lim);
else
    fprintf(2,'  %-14s = %9.5g   <= %-9.5g  VUOT %.1f%%\n', name, val, lim, (val/lim-1)*100);
end
end

function ts = lastTimeAbove(t, e, tol)
%LASTTIMEABOVE  Thoi diem CUOI CUNG ma |e| con vuot dung sai = thoi gian xac lap.
k = find(abs(e) > tol, 1, 'last');
if isempty(k), ts = 0; else, ts = t(k); end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

function [t,y] = ts2tv(simOut, varName)
t = []; y = [];
try, v = simOut.(varName); catch, return; end
if isa(v,'timeseries')
    t = v.Time(:); y = v.Data(:);
elseif isnumeric(v)
    y = v(:); t = (0:numel(y)-1).';
elseif isstruct(v) && isfield(v,'time') && isfield(v,'signals')
    t = v.time(:); y = v.signals.values(:);
end
end
