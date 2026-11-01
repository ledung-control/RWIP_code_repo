function r5_E1_S1S2()
% Round 5 E1 supplement (protocol v1.4 limitation closed): scenarios S1 (+-30% mismatch, plant only, controller nominal) and S2 (theta_dot impulse at t = 5 s,
% 1.0 and 2.0 rad/s) for the E1 final agents (seeds 1-5; seeds 3,5 NOT gate-passed) and the stored paper agents, raw and pi(0)-removed.
% Definitions copied from the authors' Compare_v13_robust.m (pert table, P.dist, recovery criterion |theta|<0.01 & |phi|<0.05 in last 10 s). Authors' files unchanged.
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(proj);
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
diary(fullfile(Ldir,sprintf('E1_S1S2_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E1_S1S2 %s %s\n', datestr(now,31), version);
P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; Ts=1e-3; Tf=60;
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;
pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
kicks = [1.0 2.0];
recovered = @(o) ~o.diverged && isfinite(o.IAEtheta) && max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && max(abs(o.phi(o.t > o.t(end)-10))) < 0.05;
newdir = {fullfile(here,'..','agents','E1_final'), fullfile(here,'..','agents','E1_final_skipbc')};
rows = {};
for seed = 1:5
  gp = 1; if any(seed==[3 5]), gp = 0; end   % gate-passed flag (seeds 3,5 re-run with SkipBCGate)
  src = {}; 
  if gp, d = newdir{1}; else, d = newdir{2}; end
  d = fullfile(d, sprintf('c3_SPH10_g0.9995_s%d',seed));
  src(end+1,:) = {fullfile(d,sprintf('agent_bconly_seed%d.mat',seed)),'BC-only','new_equals_paper'};
  src(end+1,:) = {fullfile(d,sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3','new'};
  src(end+1,:) = {fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3','paper_stored'};
  for i = 1:size(src,1)
    if ~isfile(src{i,1}), fprintf('missing %s\n', src{i,1}); continue; end
    S = load(src{i,1}); ag = S.agent; oEq = rl_obs_v13(0,0,0,0,0,0); a0 = pol(ag,oEq);
    for var = {'raw','unbiased'}
      ab = 0; if strcmp(var{1},'unbiased'), ab = a0; end
      mk = @(P) setRL(P, ag, ab, S);
      for ip = 1:size(pert,1)
        P = P0; for f = 1:numel(pert{ip,2}), P.(pert{ip,2}{f}) = P0.(pert{ip,2}{f}) * pert{ip,3}{f}; end
        o = sim_rwip_v13('PSOFLC_RL', mk(P), IC, Tf, Ts);
        dv = isfield(o,'diverged') && o.diverged;
        rows(end+1,:) = {seed,gp,src{i,3},src{i,2},var{1},'S1',pert{ip,1},NaN,dv,o.IAEtheta,o.IAEphi,o.phi_end,recovered(o)}; %#ok<AGROW>
        fprintf('s%d %-16s %-7s %-8s S1 %-9s div=%d IAEth=%.5g IAEphi=%.5g\n', seed, src{i,3}, src{i,2}, var{1}, pert{ip,1}, dv, o.IAEtheta, o.IAEphi);
      end
      for k = kicks
        P = mk(P0); P.dist = struct('t',5.0,'dthetadot',k);
        o = sim_rwip_v13('PSOFLC_RL', P, IC, Tf, Ts); dv = isfield(o,'diverged') && o.diverged;
        rows(end+1,:) = {seed,gp,src{i,3},src{i,2},var{1},'S2','kick',k,dv,o.IAEtheta,o.IAEphi,o.phi_end,recovered(o)}; %#ok<AGROW>
        fprintf('s%d %-16s %-7s %-8s S2 kick=%.1f div=%d recovered=%d IAEth=%.5g\n', seed, src{i,3}, src{i,2}, var{1}, k, dv, recovered(o), o.IAEtheta);
      end
    end
  end
  T = cell2table(rows,'VariableNames',{'Seed','GatePassed','Source','Group','Variant','Scenario','Case','Kick','Diverged','IAEtheta','IAEphi','phi_end','Recovered'});
  writetable(T, fullfile(Rdir,['E1_S1S2_' tag '.csv']));   % rewritten after every seed (resumable inspection)
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function P = setRL(P, ag, ab, S), P.rlPolicy = @(o) pol(ag,o)-ab; P.rlTs = S.cfg.Ts_agent; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax; end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
