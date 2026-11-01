function r5_E5()
% Round 5 E5 (protocol v1.0 sec. 5 with the sizes and definitions FIXED in v1.6 before running).  NEW IMPLEMENTATION (noise and delay in sim_rwip_r5.m).
% Same draws for every policy (common random numbers).  Policies: PSO-FLC, SMC-ESO (own outer loop), PSO-FLC+SMC-ESO, the 10 stored paper agents with pi(0) removed (rate as the paper: agent 100 Hz).
% Groups: IC (50 draws: theta0~U[-0.15,0.15], theta_dot0~U[-1,1], phi0~U[-2,2]); mismatch (50: mgl_eq, I_p,eq, K_t each x(1+U[-0.3,0.3]), IC = [0.10;0;0;0]);
%         noise (50: sigma_theta 1e-3 rad, sigma_phi_dot 1e-2 rad/s on the controller inputs, IC = [0.10;0;0;0]);  combined (100: IC + mismatch + noise + delay drawn from {0,5,10} ms);
%         delay (levels {0,5,10} ms x the first 20 IC draws).  Controller keeps nominal parameters (g_theta_nom).  Saturation +-24 V as in the simulator.  60 s.
% Primary outcome: recovered (not diverged, |theta|<0.01 and |phi|<0.05 in the last 10 s); secondary IAE_theta, IAE_phi.
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(here); addpath(proj);
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
outcsv = fullfile(Rdir,['E5_' tag '.csv']);
diary(fullfile(Ldir,sprintf('E5_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E5 %s %s\n', datestr(now,31), version);
P0 = make_P_v13('fis_rwip_v13_best.fis'); Ts=1e-3; Tf=60; P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;
% equivalence of the extended simulator with the authors' at defaults
for m = {'PSOFLC','SMCESO'}
  P = P0; if strcmp(m{1},'SMCESO'), P.Kp_phi_mag = P.SMC_Kp_phi_mag; P.Kv_phi_mag = P.SMC_Kv_phi_mag; end
  a = sim_rwip_v13(m{1},P,[0.1;0;0;0],10,Ts); b = sim_rwip_r5(m{1},P,[0.1;0;0;0],10,Ts);
  assert(a.IAEtheta==b.IAEtheta && a.IAEphi==b.IAEphi, 'sim_rwip_r5 changed behaviour at defaults'); fprintf('equivalence %s ok (%.15g)\n', m{1}, a.IAEtheta);
end
rng(20261031,'twister');
nIC=50; nMM=50; nNZ=50; nCB=100;
IC1 = [0.15*(2*rand(nIC,1)-1), 1*(2*rand(nIC,1)-1), 2*(2*rand(nIC,1)-1)];
MM1 = 0.3*(2*rand(nMM,3)-1);
NZ1 = 1000 + (1:nNZ).';
ICc = [0.15*(2*rand(nCB,1)-1), 1*(2*rand(nCB,1)-1), 2*(2*rand(nCB,1)-1)];
MMc = 0.3*(2*rand(nCB,3)-1); NZc = 2000 + (1:nCB).'; DLc = 5*(randi(3,nCB,1)-1);
cfg = {}; % group, draw, ic[3], mm[3], delay, noiseSeed (0 = none)
for i=1:nIC, cfg(end+1,:) = {'IC',i,IC1(i,:),[0 0 0],0,0}; end %#ok<AGROW>
for i=1:nMM, cfg(end+1,:) = {'mismatch',i,[0.10 0 0],MM1(i,:),0,0}; end %#ok<AGROW>
for i=1:nNZ, cfg(end+1,:) = {'noise',i,[0.10 0 0],[0 0 0],0,NZ1(i)}; end %#ok<AGROW>
for i=1:nCB, cfg(end+1,:) = {'combined',i,ICc(i,:),MMc(i,:),DLc(i),NZc(i)}; end %#ok<AGROW>
for L = [0 5 10], for i=1:20, cfg(end+1,:) = {sprintf('delay%d',L),i,IC1(i,:),[0 0 0],L,0}; end, end %#ok<AGROW>
fprintf('%d configurations per policy\n', size(cfg,1));
recovered = @(o) ~o.diverged && isfinite(o.IAEtheta) && max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && max(abs(o.phi(o.t > o.t(end)-10))) < 0.05;
pol = {'PSO-FLC','PSOFLC',''; 'SMC-ESO','SMCESO',''; 'PSO-FLC+SMC-ESO','PSOFLC_SMCESO',''};
for seed=1:5
  pol(end+1,:) = {sprintf('BC-only s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bconly_seed%d.mat',seed))}; %#ok<AGROW>
  pol(end+1,:) = {sprintf('BC+TD3 s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed))}; %#ok<AGROW>
end
rows = {}; done = {};
if isfile(outcsv), T0 = readtable(outcsv); rows = table2cell(T0); done = unique(T0.Policy); fprintf('resuming: %d policies done\n', numel(done)); end
for ip = 1:size(pol,1)
  if any(strcmp(done,pol{ip,1})), continue; end
  if strcmp(pol{ip,2},'PSOFLC_RL'), S = load(pol{ip,3}); ag = S.agent; a0 = polA(ag,rl_obs_v13(0,0,0,0,0,0)); end
  for ic = 1:size(cfg,1)
    P = P0; P.ctrlTs = 1e-3; P.smcTs = 1e-3; P.rlTs = 1e-2;
    if strcmp(pol{ip,2},'SMCESO'), P.Kp_phi_mag = P0.SMC_Kp_phi_mag; P.Kv_phi_mag = P0.SMC_Kv_phi_mag; end
    if strcmp(pol{ip,2},'PSOFLC_RL'), P.rlPolicy = @(o) polA(ag,o)-a0; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax; end
    mm = cfg{ic,4}; P.mgl_eq = P0.mgl_eq*(1+mm(1)); P.Ip_eq = P0.Ip_eq*(1+mm(2)); P.Kt = P0.Kt*(1+mm(3));
    P.delayMs = cfg{ic,5};
    if cfg{ic,6} > 0, P.meas = struct('sig_th',1e-3,'sig_phd',1e-2,'seed',cfg{ic,6}); end
    x0 = cfg{ic,3}; IC = [x0(1); x0(2); x0(3); 0];
    t0 = tic; o = sim_rwip_r5(pol{ip,2}, P, IC, Tf, Ts); wall = toc(t0);
    dv = isfield(o,'diverged') && o.diverged;
    rows(end+1,:) = {pol{ip,1},cfg{ic,1},cfg{ic,2},x0(1),x0(2),x0(3),mm(1),mm(2),mm(3),cfg{ic,5},cfg{ic,6},dv,recovered(o),gv(o,'IAEtheta'),gv(o,'IAEphi'),wall}; %#ok<AGROW>
  end
  T = cell2table(rows,'VariableNames',{'Policy','Group','Draw','theta0','thetad0','phi0','f_mgl','f_Ip','f_Kt','delay_ms','noise_seed','Diverged','Recovered','IAEtheta','IAEphi','wall_s'});
  writetable(T,outcsv);
  Tp = T(strcmp(T.Policy,pol{ip,1}),:); fprintf('%-16s recovered %d/%d, diverged %d\n', pol{ip,1}, sum(Tp.Recovered), height(Tp), sum(Tp.Diverged));
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function v = gv(o,f), if isfield(o,f) && ~isempty(o.(f)), v = o.(f); else, v = NaN; end, end
function a = polA(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
