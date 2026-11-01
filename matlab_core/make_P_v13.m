function P = make_P_v13(fisFile)
%MAKE_P_V13  Struct tham so dung chung cho sim_rwip_v13.
%   P = make_P_v13()                      -> dung fis_rwip_opt.fis
%   P = make_P_v13('fis_rwip_v13_best.fis')

if nargin < 1 || isempty(fisFile), fisFile = 'fis_rwip_v13_best.fis'; end

%% --- Doi tuong (giong het RWIP_Dynamics trong RWIP_PSO_FLC.slx) ---
mp=0.5297; mr=0.29225; Ip=11.57e-3; Lp=0.256; Lr=0.385; g=9.80665;
P.Ip_eq  = Ip + mp*Lp^2 + mr*Lr^2;        % 0.089606
P.Ir     = 82.1953125e-3;
P.bp     = 1e-3;    P.br = 5e-4;
P.Ra     = 2.0;     P.Ke_m = 0.5;   P.Kt = 0.5;
P.mgl_eq = g*(mp*Lp + mr*Lr);             % 2.43318

%% --- Vong ngoai cascade (DUNG CHUNG cho ca hai bo dieu khien) ---
% Luu y: o day luu DO LON; sim_rwip_v13 ap dung theta_ref = -(Kp*phi + Kv*phi_dot).
P.theta_ref_max = 0.035;

%% --- PSO-FLC v13 (nghiem CHINH THUC tu pso_rwip_v13 tren Simulink) ---
% Nguon: pso_rwip_v13_result.mat, 30 particle x 20 vong, dang co rang buoc.
%   J_best = 0.314771  va  gv = 0  (DAT DU CA 4 RANG BUOC):
%     vot lo theta = 0.0059865  <= 0.006
%     ts_theta     = 2.78 s     <= 3.0
%     phi_max      = 0.5194 rad <= 0.52
%     ts_phi       = 10.72 s    <= 11.0
% FIS tuong ung: fis_rwip_v13_best.fis  (s_e=0.20969, s_de=0.78364, s_u=0.290554)
P.fisFile   = fisFile;
P.Ke        = 1.93975;
P.Kde       = 0.678232;
P.Ku        = 23.4172;
P.Kd_theta  = 2.91123;
P.Kd_phi    = 0.190429;
P.Kp_phi_mag = 0.00437824;
P.Kv_phi_mag = 0.0183369;

%% --- SMC-ESO cascade ---
% Gain rieng cho SMC (moi bo duoc chinh dinh RIENG -- do la cach so sanh cong
% bang; ep chung gain se thien vi bo nay hay bo kia).
% w0=20 chu KHONG phai 60 nhu ban goc: voi cascade, w0=60 gay chattering ma
% banh da tich phan hai lan -> phi khong bao gio dung (Urms 7-12 V thay vi 0.19).
P.G = struct('c',8,'k',5,'eta',1.0,'phi_bl',0.05,'w0',20,'Vmax',24);
P.uCorrMax = 10;      % chi dung cho mode lai PSOFLC_SMCESO

%% --- Vong ngoai RIENG cho SMC (chinh dinh doc lap) ---
P.SMC_Kp_phi_mag = 0.12;
P.SMC_Kv_phi_mag = 0.018;
end
