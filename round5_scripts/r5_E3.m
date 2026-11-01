function r5_E3()
% Round 5 E3 (protocol v1.0 sec. 4 as corrected in v1.5): sampling-rate sensitivity of the controller comparison.  NEW IMPLEMENTATION (sim_rwip_r5.m = copy of the authors' sim with ctrlTs + LQR mode).
% Rates:  'paper' = as in the authors' evaluator (outer loop + FLC and SMC-ESO at 1 kHz, residual agent at 100 Hz ZOH);
%         'all100' = outer loop, FLC, SMC-ESO and residual agent all at 100 Hz (ZOH);  'all1k' = everything at 1 kHz (residual agent evaluated every 1 ms: deployment-rate sensitivity, agents were trained at 100 Hz).
% No retuning.  Controllers: PSO-FLC, SMC-ESO (own outer loop, as the paper), PSO-FLC+SMC-ESO, stored paper agents seeds 1-5 (BC-only, BC+TD3; pi(0) removed; raw only for S0).
% Scenarios (definitions of the authors' Compare_v13_robust.m): S0 nominal 60 s; S1 +-30 % mgl_eq, I_p,eq, K_t one at a time + adverse (plant only, controller nominal); S2 theta_dot impulse 1.0 and 2.0 rad/s at t = 5 s.
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(here); addpath(proj);
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
outcsv = fullfile(Rdir,['E3_' tag '.csv']);
diary(fullfile(Ldir,sprintf('E3_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E3 %s %s\n', datestr(now,31), version);
P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; Ts=1e-3; Tf=60;
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;
% --- equivalence check: defaults of sim_rwip_r5 reproduce sim_rwip_v13 exactly ---
for m = {'PSOFLC','SMCESO','PSOFLC_SMCESO'}
  P = P0; if strcmp(m{1},'SMCESO'), P.Kp_phi_mag = P.SMC_Kp_phi_mag; P.Kv_phi_mag = P.SMC_Kv_phi_mag; end
  a = sim_rwip_v13(m{1},P,IC,10,Ts); b = sim_rwip_r5(m{1},P,IC,10,Ts);
  fprintf('equivalence %-14s IAEtheta v13 = %.15g  r5 = %.15g  diff = %g\n', m{1}, a.IAEtheta, b.IAEtheta, a.IAEtheta-b.IAEtheta);
  assert(a.IAEtheta==b.IAEtheta && a.IAEphi==b.IAEphi, 'sim_rwip_r5 does not reproduce sim_rwip_v13 at defaults');
end
pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
kicks = [1.0 2.0];
recovered = @(o) ~o.diverged && isfinite(o.IAEtheta) && max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && max(abs(o.phi(o.t > o.t(end)-10))) < 0.05;
rates = {'paper','all100','all1k'};
% controller list: {name, mode, agentFile, variant, scenariosAll}
ctrl = {'PSO-FLC','PSOFLC','', 'none', true; 'SMC-ESO','SMCESO','', 'none', true; 'PSO-FLC+SMC-ESO','PSOFLC_SMCESO','', 'none', true};
for seed = 1:5
  ctrl(end+1,:) = {sprintf('BC-only s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bconly_seed%d.mat',seed)),'unbiased',true}; %#ok<AGROW>
  ctrl(end+1,:) = {sprintf('BC+TD3 s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed)),'unbiased',true}; %#ok<AGROW>
  ctrl(end+1,:) = {sprintf('BC-only s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bconly_seed%d.mat',seed)),'raw',false}; %#ok<AGROW>
  ctrl(end+1,:) = {sprintf('BC+TD3 s%d',seed),'PSOFLC_RL',fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed)),'raw',false}; %#ok<AGROW>
end
done = {}; rows = {};
if isfile(outcsv), T0 = readtable(outcsv); rows = table2cell(T0); done = unique(strcat(T0.Rate,'|',T0.Controller,'|',T0.Variant)); fprintf('resuming: %d blocks done\n', numel(done)); end
for ir = 1:numel(rates)
  rate = rates{ir};
  for ic = 1:size(ctrl,1)
    key = [rate '|' ctrl{ic,1} '|' ctrl{ic,4}]; if any(strcmp(done,key)), continue; end
    cases = {}; for ip=1:size(pert,1), cases(end+1,:) = {'S1',pert{ip,1},NaN,ip}; end %#ok<AGROW>
    if ctrl{ic,5}, for k=kicks, cases(end+1,:) = {'S2','kick',k,0}; end, else, cases = cases(1,:); end %#ok<AGROW>
    for icase = 1:size(cases,1)
      P = P0;
      switch rate
        case 'paper',  P.ctrlTs = 1e-3; P.smcTs = 1e-3; P.rlTs = 1e-2;
        case 'all100', P.ctrlTs = 1e-2; P.smcTs = 1e-2; P.rlTs = 1e-2;
        case 'all1k',  P.ctrlTs = 1e-3; P.smcTs = 1e-3; P.rlTs = 1e-3;
      end
      if strcmp(ctrl{ic,2},'SMCESO'), P.Kp_phi_mag = P0.SMC_Kp_phi_mag; P.Kv_phi_mag = P0.SMC_Kv_phi_mag; end
      if strcmp(cases{icase,1},'S1')
        ip = cases{icase,4}; for f = 1:numel(pert{ip,2}), P.(pert{ip,2}{f}) = P0.(pert{ip,2}{f}) * pert{ip,3}{f}; end
      else, P.dist = struct('t',5.0,'dthetadot',cases{icase,3}); end
      if strcmp(ctrl{ic,2},'PSOFLC_RL')
        S = load(ctrl{ic,3}); ag = S.agent; a0 = pol(ag,rl_obs_v13(0,0,0,0,0,0)); ab = 0; if strcmp(ctrl{ic,4},'unbiased'), ab = a0; end
        P.rlPolicy = @(o) pol(ag,o)-ab; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax;
      end
      t0 = tic; o = sim_rwip_r5(ctrl{ic,2}, P, IC, Tf, Ts); wall = toc(t0);
      dv = isfield(o,'diverged') && o.diverged;
      rows(end+1,:) = {rate,ctrl{ic,1},ctrl{ic,4},cases{icase,1},cases{icase,2},cases{icase,3},dv,gv(o,'IAEtheta'),gv(o,'IAEphi'),gv(o,'OS_theta'),gv(o,'ts_theta'),gv(o,'ts_phi'),gv(o,'Urms'),gv(o,'phi_end'),recovered(o),wall}; %#ok<AGROW>
      fprintf('%-6s %-18s %-8s %s %-9s k=%g div=%d IAEth=%.5g IAEphi=%.5g (%.0f s)\n', rate, ctrl{ic,1}, ctrl{ic,4}, cases{icase,1}, cases{icase,2}, cases{icase,3}, dv, gv(o,'IAEtheta'), gv(o,'IAEphi'), wall);
    end
    T = cell2table(rows,'VariableNames',{'Rate','Controller','Variant','Scenario','Case','Kick','Diverged','IAEtheta','IAEphi','OS_theta','ts_theta','ts_phi','Urms','phi_end','Recovered','wall_s'});
    writetable(T,outcsv);
  end
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function v = gv(o,f), if isfield(o,f) && ~isempty(o.(f)), v = o.(f); else, v = NaN; end, end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
