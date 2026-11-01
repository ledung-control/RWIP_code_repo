%SIM_RWIP_R5  Round 5 NEW IMPLEMENTATION = copy of the authors' sim_rwip_v13.m (unchanged file) with two additions, both inactive by default:
%   P.ctrlTs : sample time of the outer loop + base inner loop (FLC / LQR); zero-order hold in between. Default = Ts (identical to sim_rwip_v13).
%   mode 'LQR' : inner loop V = -K*[theta - theta_ref; theta_dot; phi_dot] with K = P.lqrK (1x3); same outer loop as the other modes.
%   (Authors' P.smcTs and P.rlTs are used unchanged for SMC-ESO and the residual agent.)  Equivalence with sim_rwip_v13 at defaults is checked in r5_E3.m.
function out = sim_rwip_r5(mode, P, IC, Tf, Ts)
%SIM_RWIP_V13  Mo phong RWIP (RK4, buoc co dinh) cho CA HAI bo dieu khien
%              tren CUNG doi tuong va CUNG vong ngoai cascade.
%
%   out = sim_rwip_v13(mode, P, IC, Tf, Ts)
%
%   mode : 'PSOFLC'         -- PSO-FLC v13 (mo so + phan hoi trang thai)
%          'SMCESO'         -- SMC-ESO cascade doc lap (SMCESO_cascade.m)
%          'PSOFLC_SMCESO'  -- lai: PSO-FLC lam nen + SMC-ESO lam lop hieu chinh
%   P    : struct tham so (dung make_P_v13 de tao)
%   IC   : [theta0; theta_dot0; phi0; phi_dot0]
%   Tf   : thoi gian mo phong (s)
%   Ts   : buoc thoi gian (s), nen 1e-3
%
%   out  : struct co t, theta, theta_dot, phi, phi_dot, u, theta_ref, s
%          va cac chi so: IAEtheta IAEphi IAEu RMSEtheta RMSEphi Urms
%                         OS_theta ts_theta ts_phi phi_end phidot_end phi_max
%                         Eu Edu IntSinTheta
%
% CA BA mode dung CHUNG vong ngoai:
%     theta_ref = sat( -(Kp_phi*phi + Kv_phi*phi_dot), +-theta_ref_max )
% nen so sanh giua chung co lap dung luat vong trong.

if nargin < 5, Ts = 1e-3; end
if nargin < 4, Tf = 600;   end
if nargin < 3, IC = [0.10; 0; 0; 0]; end

% --- tham so co hoc CUA DOI TUONG (co the da bi nhieu loan) ---
Ip=P.Ip_eq; Ir=P.Ir; bp=P.bp; br=P.br;
Ra=P.Ra; Kem=P.Ke_m; Kt=P.Kt; mgl=P.mgl_eq;

% --- he so control-affine MA BO DIEU KHIEN GIA DINH ---
% Trong thu nghiem ben vung, DOI TUONG bi nhieu loan nhung BO DIEU KHIEN van
% giu gia tri DANH NGHIA (no khong biet doi tuong da doi). Neu dung P.Ip_eq /
% P.Kt da nhieu loan o day thi hoa ra bo dieu khien duoc "mach nuoc" sai lech
% -- phep thu ben vung mat het y nghia.
if isfield(P,'g_theta_nom') && ~isempty(P.g_theta_nom)
    g_theta = P.g_theta_nom;
else
    g_theta = -(Kt/Ra)/Ip;
end

% --- nhieu ngoai: xung van toc tren theta_dot ---
% P.dist = struct('t', 5.0, 'dthetadot', 0.30)
hasDist = isfield(P,'dist') && ~isempty(P.dist) && isfield(P.dist,'dthetadot');
if hasDist, tDist = P.dist.t; dKick = P.dist.dthetadot; distDone = false;
else,       tDist = inf;      dKick = 0;                distDone = true; end

useFLC = any(strcmpi(mode,{'PSOFLC','PSOFLC_SMCESO','PSOFLC_RL'}));
useLQR = strcmpi(mode,'LQR');
ctrlTs = getfd(P,'ctrlTs',Ts); ctrlEvery = max(1, round(ctrlTs/Ts)); tref_h = 0; Vb_h = 0;
useSMC = any(strcmpi(mode,{'SMCESO','PSOFLC_SMCESO'}));
useRL  = strcmpi(mode,'PSOFLC_RL');
assert(useFLC||useSMC||useLQR,'mode phai la PSOFLC | SMCESO | PSOFLC_SMCESO | PSOFLC_RL');

% --- Lop hieu chinh hoc duoc (danh gia chinh sach da huan luyen) ---
% P.rlPolicy : function handle  u_corr = f(obs),  obs la vector 6x1 da chuan hoa
% P.rlTs     : chu ky goi agent (mac dinh 0.01 s), giu bac khong giua cac lan
% P.rlUmax   : bien hanh dong (V), mac dinh 10
if useRL
    assert(isfield(P,'rlPolicy') && ~isempty(P.rlPolicy), ...
        'mode PSOFLC_RL can P.rlPolicy (function handle u_corr = f(obs)).');
    rlTs   = getfd(P,'rlTs',   0.01);
    rlUmax = getfd(P,'rlUmax', 10);
    rlEvery = max(1, round(rlTs/Ts));
    u_rl = 0;
else
    rlEvery = inf; u_rl = 0; rlUmax = 0;
end

% ===========================================================================
% P.smcTs -- CHU KY CAP NHAT SMC-ESO (giu bac khong giua cac lan). Them 08/08.
% ---------------------------------------------------------------------------
% MAC DINH = Ts, tuc SMC-ESO tinh moi buoc tich phan: hanh vi Y HET ban cu.
% Bang 1-6 va moi ket qua da cong bo KHONG bi anh huong.
%
% VI SAO CAN: lop hieu chinh RL chay ZOH 100 Hz (P.rlTs), nhung tham chieu
% SMC-ESO chay 1 kHz. So hai hang do voi nhau la so mot bo 1 kHz voi mot bo
% 100 Hz -- chenh lech thu duoc TRON hai thu khac han nhau:
%     (a) sai so hoi quy cua behavior cloning        <- cai ta muon do
%     (b) cai gia cua viec chay o 100 Hz thay vi 1 kHz  <- khong lien quan BC
% Do duoc o hat giong 1: d rms = +57% -- khong the biet bao nhieu la (a).
%
% Dat P.smcTs = P.rlTs cho phep chay CHINH SMC-ESO o dung dieu kien trien khai
% cua agent, tao ra hang tham chieu thu ba tach bach duoc (a) khoi (b).
%
% Ts truyen vao SMCESO_cascade cung phai la smcTs: ESO tich phan tren chu ky
% that giua hai lan goi, khong phai tren buoc tich phan doi tuong. Dat sai cho
% nay se cho mot bo ESO chinh dinh nham chu ky -- dung loai loi ngam ma du an
% nay da vap ba lan.
% ===========================================================================
smcTs    = getfd(P, 'smcTs', Ts);
smcEvery = max(1, round(smcTs/Ts));
Vs_hold  = 0;  s_hold = 0;

% --- Bang tra mat dieu khien mo (nhanh hon evalfis ~1000 lan) ---
if useFLC
    persistent SURF SURFKEY GRD
    % P.surfN cho phep doi do phan giai bang tra (dung trong
    % Benchmark_ComputeTime_v13 de do anh huong len chat luong dieu khien).
    if isfield(P,'surfN') && ~isempty(P.surfN), nG = P.surfN; else, nG = 201; end
    key = sprintf('%s|%d', P.fisFile, nG);     % cache theo CA file VA do phan giai
    if isempty(SURFKEY) || ~strcmp(SURFKEY,key)
        fis = readfis(P.fisFile);
        GRD = linspace(-1,1,nG);
        [A,B] = ndgrid(GRD,GRD);
        SURF = reshape(evalfis(fis,[A(:) B(:)]), nG, nG);
        SURFKEY = key;
    end
end

N = round(Tf/Ts);
t = (0:N).'*Ts;
TH=zeros(N+1,1); THD=zeros(N+1,1); PH=zeros(N+1,1); PHD=zeros(N+1,1);
U =zeros(N+1,1); TR =zeros(N+1,1); S =zeros(N+1,1);
UC=zeros(N+1,1);      % u_corr cua lop hieu chinh (SMC hoac RL), de tinh ty le rho

% Ghi lai (quan sat, hanh dong) o TAN SO AGENT de sinh du lieu bat chuoc (BC).
% Dung CHUNG rl_obs_v13 voi moi truong huan luyen -- khong co ban sao nao de lech.
doLog = isfield(P,'logObs') && ~isempty(P.logObs) && P.logObs;
logEvery = max(1, round(getfd(P,'rlTs',0.01)/Ts));
if doLog
    nLog = floor(N/logEvery) + 1;
    OBSLOG = zeros(6, nLog);  ALOG = zeros(1, nLog);  iLog = 0;
end

x = IC(:);
z = [x(1)-0; 0; 0];        % trang thai ESO tren bien sai so e
TH(1)=x(1); THD(1)=x(2); PH(1)=x(3); PHD(1)=x(4);

for n = 1:N
    th=x(1); thd=x(2); ph=x(3); phd=x(4);

    % ---- VONG NGOAI + VONG TRONG CO SO (lay mau moi ctrlEvery buoc; mac dinh moi buoc) ----
    if mod(n-1, ctrlEvery) == 0
        tref_h = -(P.Kp_phi_mag*ph + P.Kv_phi_mag*phd);
        tref_h = max(-P.theta_ref_max, min(P.theta_ref_max, tref_h));
        Vb_h = 0;
        if useFLC
            i1 = max(-1,min(1, P.Ke *(th - tref_h)));
            i2 = max(-1,min(1, P.Kde*thd));
            Vb_h = P.Ku*interp2lin(GRD, SURF, i1, i2) + P.Kd_theta*thd + P.Kd_phi*phd;
        end
        if useLQR
            Vb_h = -(P.lqrK(1)*(th - tref_h) + P.lqrK(2)*thd + P.lqrK(3)*phd);
        end
    end
    tref = tref_h;  TR(n) = tref;
    V = Vb_h; sVal = 0;
    if useSMC
        % Giu bac khong: chi tinh lai SMC-ESO moi smcEvery buoc. Voi mac dinh
        % smcTs = Ts thi smcEvery = 1 -> tinh moi buoc, y het ban cu.
        if mod(n-1, smcEvery) == 0
            if strcmpi(mode,'PSOFLC_SMCESO')
                [Vs_hold, z, s_hold] = SMCESO_cascade(th, tref, z, g_theta, smcTs, ...
                                    setfield(P.G, 'Vmax', P.uCorrMax)); %#ok<SFLD>
            else
                [Vs_hold, z, s_hold] = SMCESO_cascade(th, tref, z, g_theta, smcTs, P.G);
            end
        end
        V = V + Vs_hold;
        UC(n) = Vs_hold;
        sVal  = s_hold;
    end
    if useRL
        % Giu bac khong: chi goi chinh sach moi rlEvery buoc
        if mod(n-1, rlEvery) == 0
            obs  = rl_obs_v13(th, thd, ph, phd, tref, V);
            a    = P.rlPolicy(obs);
            u_rl = rlUmax * max(-1, min(1, a(1)));
        end
        V = V + u_rl;
        UC(n) = u_rl;
    end
    if doLog && mod(n-1, logEvery) == 0
        iLog = iLog + 1;
        % Quan sat dung bang luc lop hieu chinh "nhin thay": u_FLC la phan NEN
        % (da tru u_corr), giong het trong moi truong huan luyen.
        OBSLOG(:,iLog) = rl_obs_v13(th, thd, ph, phd, tref, V - UC(n));
        ALOG(iLog)     = UC(n) / max(getfd(P,'uCorrMax',10), eps);
    end

    V = max(-24, min(24, V));
    U(n)=V; S(n)=sVal;

    % ---- RK4 tren doi tuong ----
    f = @(X) rwip_deriv(X, V, Ip,Ir,bp,br,Ra,Kem,Kt,mgl);
    k1=f(x); k2=f(x+Ts/2*k1); k3=f(x+Ts/2*k2); k4=f(x+Ts*k3);
    x = x + Ts/6*(k1+2*k2+2*k3+k4);

    % Nhieu ngoai: cong mot lan vao theta_dot khi vuot moc thoi gian
    if ~distDone && (n*Ts) >= tDist
        x(2) = x(2) + dKick;  distDone = true;
    end

    if ~all(isfinite(x)) || abs(x(1))>1.5
        TH=TH(1:n); THD=THD(1:n); PH=PH(1:n); PHD=PHD(1:n);
        U=U(1:n); TR=TR(1:n); S=S(1:n); UC=UC(1:n); t=t(1:n);
        out = pack(t,TH,THD,PH,PHD,U,TR,S,UC,true); return
    end
    TH(n+1)=x(1); THD(n+1)=x(2); PH(n+1)=x(3); PHD(n+1)=x(4);
end
U(end)=U(end-1); TR(end)=TR(end-1); S(end)=S(end-1); UC(end)=UC(end-1);
out = pack(t,TH,THD,PH,PHD,U,TR,S,UC,false);
if doLog, out.obsLog = OBSLOG(:,1:iLog); out.aLog = ALOG(1:iLog); end
end

%% ------------------------------------------------------------------
function v = getfd(S, f, d)
if isfield(S,f) && ~isempty(S.(f)), v = S.(f); else, v = d; end
end

%% ------------------------------------------------------------------
function dx = rwip_deriv(x, V, Ip,Ir,bp,br,Ra,Kem,Kt,mgl)
th=x(1); thd=x(2); phd=x(4);
tau  = (Kt/Ra)*(V - Kem*phd);
thdd = (-tau + br*phd + mgl*sin(th) - bp*thd)/Ip;
phdd = (tau - br*phd)/Ir - thdd;
dx = [thd; thdd; phd; phdd];
end

%% ------------------------------------------------------------------
function v = interp2lin(g, S, a, b)
% Noi suy song tuyen tren luoi deu [-1,1] -- nhanh, khong goi interp2.
n = numel(g); h = 2/(n-1);
xa = (a+1)/h; xb = (b+1)/h;
i = min(max(floor(xa),0), n-2); j = min(max(floor(xb),0), n-2);
fa = xa-i; fb = xb-j; i=i+1; j=j+1;
v = S(i,j)*(1-fa)*(1-fb) + S(i+1,j)*fa*(1-fb) ...
  + S(i,j+1)*(1-fa)*fb   + S(i+1,j+1)*fa*fb;
end

%% ------------------------------------------------------------------
function out = pack(t,TH,THD,PH,PHD,U,TR,S,UC,diverged)
out.t=t; out.theta=TH; out.theta_dot=THD; out.phi=PH; out.phi_dot=PHD;
out.u=U; out.theta_ref=TR; out.s=S; out.u_corr=UC; out.diverged=diverged;
% rho = ty le can thiep cua lop hieu chinh so voi bo nen. Dai lop hieu chinh
% "residual" that su: SMC-ESO do duoc rho in [0.28, 0.43].
if ~diverged && any(UC~=0)
    out.u_corr_max = max(abs(UC));
    out.u_corr_rms = sqrt(mean(UC.^2));
    out.rho = max(abs(UC)) / max(max(abs(U - UC)), eps);
else
    out.u_corr_max = 0; out.u_corr_rms = 0; out.rho = 0;
end
if diverged
    fn={'IAEtheta','IAEphi','IAEu','RMSEtheta','RMSEphi','Urms','OS_theta', ...
        'ts_theta','ts_phi','phi_end','phidot_end','phi_max','Eu','Edu','IntSinTheta'};
    for k=1:numel(fn), out.(fn{k}) = NaN; end
    return
end
dt = t(2)-t(1);
out.IAEtheta = trapz(t,abs(TH));   out.IAEphi = trapz(t,abs(PH));
out.IAEu     = trapz(t,abs(U));
out.RMSEtheta= sqrt(mean(TH.^2));  out.RMSEphi= sqrt(mean(PH.^2));
out.Urms     = sqrt(mean(U.^2));
% Vot lo THUC = phan di qua diem dat, phia doi dien voi diem xuat phat
if TH(1) >= 0, out.OS_theta = max(0,-min(TH));
else,          out.OS_theta = max(0, max(TH)); end
kt = find(abs(TH)>0.002,1,'last'); out.ts_theta = tern(isempty(kt),0,t(max(kt,1)));
kp = find(abs(PH)>0.020,1,'last'); out.ts_phi   = tern(isempty(kp),0,t(max(kp,1)));
out.phi_end = PH(end); out.phidot_end = PHD(end); out.phi_max = max(abs(PH));
du = [0; diff(U)]/dt;
out.Eu = trapz(t,U.^2);  out.Edu = trapz(t,du.^2);
% Bat bien bao toan:  Int sin(theta) dt = (Ir*phi_dot(inf) - bp*theta0)/mgl_eq
out.IntSinTheta = trapz(t,sin(TH));
end
function y = tern(c,a,b), if c, y=a; else, y=b; end, end
