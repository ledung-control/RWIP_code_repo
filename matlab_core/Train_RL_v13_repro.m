function [agent, S] = Train_RL_v13_repro(seed, opts)
%TRAIN_RL_V13_REPRO  Tai lap CHINH XAC cau hinh da sinh ra agent tot nhat.
%
%   [agent, S] = Train_RL_v13_repro(1)      -- hat giong 1
%   [agent, S] = Train_RL_v13_repro(2)      -- hat giong 2 (de co sai so)
%
% ===========================================================================
% CAU HINH NAY LAY TU DAU
% ---------------------------------------------------------------------------
% Truy nguoc tu rl_v13_bc/agent_bc_seed1.mat (08/08/2026 10:04) bang
% Inspect_Agent_v13. Do la agent DUY NHAT vuot ca PSO-FLC lan SMC-ESO tren
% giao thuc danh gia that (Eval_RL_v13_final):
%
%     nhieu 1.0 rad/s : IAEtheta 0.11222 (SMC-ESO 0.12585, PSO-FLC 0.12875)
%                       ts_theta   9.388 (SMC-ESO  15.971, PSO-FLC  16.537)
%     nhieu 2.0 rad/s : IAEtheta 0.24520 (SMC-ESO 0.27652, PSO-FLC 0.30285)
%                       ts_theta  19.040 (SMC-ESO  21.168, PSO-FLC  21.862)
%     rho 0.379 -- nam trong dai SMC-ESO 0.28..0.43, VAN la lop residual
%
% Script nay GHIM toan bo tham so DUOC DO LAI tu file do, khong phu thuoc vao
% bat ky ban va nao trong rwip_env_v13.m / Train_RL_v13_bc.m sau ngay 08/08.
%
% ---------------------------------------------------------------------------
% PHAT HIEN QUAN TRONG NHAT: ActorLR = 1e-5 KHONG SAI. CHAY DAI MOI SAI.
% ---------------------------------------------------------------------------
% Ghi chu thiet ke trong Train_RL_v13_bc.m ket toi 1e-5:
%     "LR = 1e-5 : 400 ep x 2000 x 1e-5 = 8.00 don vi dich chuyen -> DA SUP"
% va vi the ha xuong 1e-6. Nhung agent thang cuoc DUNG 1e-5 -- chi voi 18
% episode tinh chinh:
%
%     LR 1e-5 x 400 ep x 2000 = 8.00 don vi  -> pha nat diem xuat phat BC
%     LR 1e-5 x  18 ep x 2000 = 0.36 don vi  <- AGENT THANG CUOC
%     LR 1e-6 x 400 ep x 2000 = 0.80 don vi  -> ban va cua ho
%
% Dai luong quyet dinh la TONG DICH CHUYEN THAM SO, khong phai learning rate.
% 0.36 don vi du de cai thien nhung chua du de pha chinh sach BC. Ket luan
% "1e-5 sup do" quy trach nhiem cho LR trong khi thu pham la SO EPISODE.
%
% => DUNG TANG MaxEpisodes voi hy vong tot hon. Cau hinh nay hoat dong CHINH VI
%    no dung som. Neu muon chay dai hon, PHAI ha LR theo ty le nghich de giu
%    tong dich chuyen ~0.36 (xem opts.KeepDrift).
%
% ---------------------------------------------------------------------------
% BON DIEM KHAC voi cac ban va sau nay -- GIU NGUYEN BAN GOC, co chu dich
% ---------------------------------------------------------------------------
%   DiscountFactor    0.995  (khong phai 0.99 hay 0.999 -- ca hai deu da thu, deu kem)
%   GradientThreshold Inf    (KHONG cat gradient)
%   mang              256    (khong phai 128)
%   uMax              10 V   (khong phai 8 -- xem canh bao cuoi file)
%   ham thuong        control-variate GOC: r = (cBase - cA)*Ts_agent
%                     KHONG bonus/penalty nguong, KHONG rScale, KHONG phiCap
% ===========================================================================

if nargin < 1 || isempty(seed), seed = 1; end
if nargin < 2, opts = struct(); end
def = struct('BCEpisodes',60, 'BCIters',3000, 'TailFrac',0.15, ...
             'WarmupEpisodes',20, 'MaxEpisodes',18, 'ActorLR',1e-5, ...
             'SaveDir','rl_v13_repro', 'KeepDrift',true, 'Plots','training-progress', ...
             'SkipBCGate',false);   % chi bat khi DO PHAN BO qua nhieu hat giong
fn = fieldnames(def);
for k = 1:numel(fn), if ~isfield(opts,fn{k}), opts.(fn{k}) = def.(fn{k}); end, end

% Giu TONG DICH CHUYEN = 0.36 don vi neu nguoi dung doi MaxEpisodes.
driftRef = 18 * 2000 * 1e-5;
if opts.KeepDrift && opts.MaxEpisodes ~= 18
    opts.ActorLR = driftRef / (opts.MaxEpisodes * 2000);
    fprintf(2, ['(!) MaxEpisodes = %d (khac 18). KeepDrift = true nen ActorLR tu\n' ...
                '    dieu chinh ve %.2e de giu tong dich chuyen = %.2f don vi.\n' ...
                '    Dat KeepDrift = false neu ban CO Y muon dich chuyen khac.\n'], ...
            opts.MaxEpisodes, opts.ActorLR, driftRef);
end

rng(seed,'twister');
if ~isfolder(opts.SaveDir), mkdir(opts.SaveDir); end

%% ---------------- Moi truong: DUNG cfg da truy nguoc ----------------
cfg0 = struct('p_ic',0.0, 'p_kick',1.0, 'p_param',0.0, ...
              'Tf',40, 'Ts',1e-3, 'Ts_agent',0.01, ...
              'uMax',10, 'thetaFail',0.5, 'failPen',200, ...
              'th_s',0.010, 'ph_s',0.100, 'w_ph',0.054, 'w_u',0.05, 'w_du',0.02, ...
              'kick_t',[3 10], 'kick_amp',[0.5 3.0], ...
              'rewardMode','cv', 'rScale',1, ...       % ham thuong GOC
              'r_bonus',0, 'r_penalty',0, ...          % KHONG so hang thua
              'phiCap',inf, 'k_alive',0, 'k_fail',0, 'failPenAuto',false);
[env, cfg] = rwip_env_v13(cfg0);
nStepEp = round(cfg.Tf/cfg.Ts_agent);
obsInfo = getObservationInfo(env);  actInfo = getActionInfo(env);
nObs = obsInfo.Dimension(1);

fprintf('=== Train_RL_v13_repro, hat giong %d ===\n', seed);
fprintf('  Tai lap cau hinh cua agent_bc_seed1.mat (08/08 10:04)\n');
fprintf('  gamma 0.995 | ActorLR %.0e | GradThresh Inf | mang [256 256]\n', opts.ActorLR);
fprintf('  uMax %g V | Tf %g s | ham thuong control-variate GOC\n', cfg.uMax, cfg.Tf);
fprintf('  %d ep BC | %d ep ham nong | %d ep tinh chinh  (tong dich chuyen %.2f)\n\n', ...
        opts.BCEpisodes, opts.WarmupEpisodes, opts.MaxEpisodes, ...
        opts.MaxEpisodes*2000*opts.ActorLR);

%% ---------------- BC + agent: dung lai Train_RL_v13_bc ----------------
% Train_RL_v13_bc da co san toan bo phan BC (sinh nhan, huan luyen actor, cong
% kiem tra). Goi lai no thay vi chep code -- nhung PHAI truyen cfg va tuy chon
% da ghim o tren. Neu ban da sua Train_RL_v13_bc sau 08/08 thi cac ban va do
% se anh huong; xem canh bao cuoi.
warning('Train_RL_v13_repro:useBC', ...
    ['Script nay goi Train_RL_v13_bc de lam phan BC. Neu file do da bi sua sau\n' ...
     '08/08 (cac ban va 2-11), ket qua co the khac ban goc. Doi chieu muc [5]\n' ...
     'cua Inspect_Agent_v13 tren agent moi de xac nhan cfg trung khop.']);

% Goi thang Train_RL_v13_bc voi TOAN BO tham so ghim qua opts -- khong can sua
% tay dong nao. Ba tuy chon DiscountFactor / GradThresh / RewardCfg da duoc
% them vao Train_RL_v13_bc.m ngay 09/08 dung cho muc dich nay.
tOpts = struct( ...
    'BCEpisodes',      opts.BCEpisodes, ...
    'BCIters',         opts.BCIters, ...
    'TailFrac',        opts.TailFrac, ...
    'WarmupEpisodes',  opts.WarmupEpisodes, ...
    'MaxEpisodes',     opts.MaxEpisodes, ...
    'WarmupChunk',     opts.WarmupEpisodes, ...   % mot khoi -> khong chia nho
    'ActorLRFineTune', opts.ActorLR, ...
    'DiscountFactor',  0.995, ...                 % <- ban goc
    'GradThresh',      Inf, ...                   % <- ban goc: KHONG cat gradient
    'RewardCfg',       cfg0, ...                  % <- ham thuong control-variate GOC
    'SkipCriticGate',  true, ...                  % cong critic hieu chuan cho ham thuong KHAC
    'SkipB1Gate',      true, ...                  % xem ly do ngay duoi
    'SkipBCGate',      opts.SkipBCGate, ...       % mac dinh false; bat de do phan bo
    'SaveDir',         opts.SaveDir, ...
    'Plots',           opts.Plots);

fprintf('Goi Train_RL_v13_bc voi cau hinh da ghim...\n\n');
agent = Train_RL_v13_bc(seed, tOpts);
S = struct('cfg',cfg, 'opts',opts, 'tOpts',tOpts, 'seed',seed);

%% ---------------- Canh bao phai doc ----------------
fprintf(2, ['\n(!) uMax = 10 V, KHAC dac ta 8 V trong ghi chu thiet ke.\n' ...
            '    Agent goc bao hoa o 10 V (max|u_corr| = 10.00, trung vi 7.08).\n' ...
            '    rho = 0.379 van nam trong dai SMC-ESO 0.28..0.43 nen ve TY LE\n' ...
            '    CAN THIEP no dung la residual. Trong bai phai neu ro bien 10 V,\n' ...
            '    hoac huan luyen lai o 8 V va bao cao lai.\n']);
end
