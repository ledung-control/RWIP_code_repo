function r5_baseline(stage)
% r5_baseline('pilot' | 'A' | 'B')   -- Round 5 Phase 1 re-run of baselines in MATLAB.
%  pilot : 3 controllers, Tf = 5 s (speed/sanity check only; not a result)
%  A     : PSO-FLC, SMC-ESO cascade, PSO-FLC+SMC-ESO, nominal IC, Tf = 60 s and 600 s
%  B     : 10 stored agents (BC-only s1-5, BC+TD3 s1-5), raw and with pi(0) subtracted (aBias), Tf = 60 s and 600 s
% Read-only on all original files. Writes ONLY to Revision_Round5\results and \logs.
% Uses the authors' own sim_rwip_v13.m, make_P_v13.m, rl_obs_v13.m, agent .mat files (unchanged).
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here)));
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs');
cd(proj); addpath(proj);
tag = ['MATLAB_' strrep(version('-release'),' ','')];
diary(fullfile(Ldir,sprintf('baseline_%s_%s_%s.txt',stage,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_baseline(%s)  %s  %s  cwd=%s\n', stage, datestr(now,31), version, pwd);
fprintf('file dates: sim_rwip_v13.m %s | make_P_v13.m %s | Eval_RL_v13_final.m %s\n', ...
  getfdate('sim_rwip_v13.m'), getfdate('make_P_v13.m'), getfdate('Eval_RL_v13_final.m'));
P0 = make_P_v13('fis_rwip_v13_best.fis'); P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;
IC = [0.10;0;0;0]; Ts = 1e-3;
fields = {'IAEtheta','IAEphi','IAEu','RMSEtheta','RMSEphi','Urms','OS_theta','ts_theta','ts_phi','phi_end','phidot_end','phi_max','Eu','Edu','IntSinTheta','rho'};
rows = {};
switch stage
  case 'pilot', Tfs = 5; case {'A','B'}, Tfs = [60 600]; otherwise, error('stage');
end
if any(strcmp(stage,{'pilot','A'}))
  modes = {'PSOFLC','SMCESO','PSOFLC_SMCESO'};
  for Tf = Tfs
    for m = 1:numel(modes)
      Pm = P0; if strcmp(modes{m},'SMCESO'), Pm.Kp_phi_mag = Pm.SMC_Kp_phi_mag; Pm.Kv_phi_mag = Pm.SMC_Kv_phi_mag; end  % same as applyOuter() in Compare_v13_robust.m
      t0 = tic; o = sim_rwip_v13(modes{m}, Pm, IC, Tf, Ts); dt = toc(t0);
      rows(end+1,:) = [{modes{m}, 'nominal', Tf, dt, isfield(o,'diverged') && o.diverged}, getm(o,fields)]; %#ok<AGROW>
      fprintf('  %-14s Tf=%4d  %6.1fs  IAEtheta=%.7g IAEphi=%.7g OS=%.7g\n', modes{m}, Tf, dt, getv(o,'IAEtheta'), getv(o,'IAEphi'), getv(o,'OS_theta'));
      save_rows(rows, fields, fullfile(Rdir, sprintf('baseline_%s_v2_%s.csv', stage, tag)));
    end
  end
else
  agentDir = fullfile(proj,'rl_v13_repro'); L = {};
  for s = 1:5, L(end+1,:) = {sprintf('BC-only_s%d',s), fullfile(agentDir,sprintf('agent_bconly_seed%d.mat',s))}; end %#ok<AGROW>
  for s = 1:5, L(end+1,:) = {sprintf('BC+TD3_s%d',s),  fullfile(agentDir,sprintf('agent_bc_seed%d.mat',s))}; end %#ok<AGROW>
  oEq = rl_obs_v13(0,0,0,0,0,0);
  for i = 1:size(L,1)
    S = load(L{i,2}); assert(isfield(S,'agent') && isfield(S,'cfg'));
    ag = S.agent; Ta = S.cfg.Ts_agent; uM = S.cfg.uMax;
    a0 = getAction(ag,{oEq(:)}); if iscell(a0), a0=a0{1}; end; a0 = double(a0(1));
    storedBias = 0; if isfield(S,'aBias') && ~isempty(S.aBias), storedBias = double(S.aBias); end
    fprintf('%s: pi(0)=%+.6f (%+.4f V), stored aBias=%g, Ts_agent=%g, uMax=%g\n', L{i,1}, a0, a0*uM, storedBias, Ta, uM);
    for variant = {'raw','unbiased'}
      ab = 0; if strcmp(variant{1},'unbiased'), ab = a0; end
      P = P0; P.rlPolicy = @(o) polAct(ag,o) - ab; P.rlTs = Ta; P.rlUmax = uM; P.uCorrMax = uM;
      for Tf = Tfs
        t0 = tic; o = sim_rwip_v13('PSOFLC_RL', P, IC, Tf, Ts); dt = toc(t0);
        rows(end+1,:) = [{L{i,1}, variant{1}, Tf, dt, isfield(o,'diverged') && o.diverged}, getm(o,fields)]; %#ok<AGROW>
        fprintf('  %-12s %-8s Tf=%4d %6.1fs IAEtheta=%.7g IAEphi=%.7g phi_end=%.5g\n', L{i,1}, variant{1}, Tf, dt, getv(o,'IAEtheta'), getv(o,'IAEphi'), getv(o,'phi_end'));
        save_rows(rows, fields, fullfile(Rdir, sprintf('baseline_B_%s.csv', tag)));
      end
    end
  end
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function a = polAct(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
function v = getv(o,f), if isfield(o,f) && ~isempty(o.(f)), v = double(o.(f)(1)); else, v = NaN; end, end
function c = getm(o,F), c = cell(1,numel(F)); for k=1:numel(F), c{k} = getv(o,F{k}); end, end
function d = getfdate(f), x = dir(which(f)); if isempty(x), d='(missing)'; else, d = x(1).date; end, end
function save_rows(rows, F, fn)
T = cell2table(rows,'VariableNames',[{'Controller','Variant','Tf','wall_s','diverged'}, F]);
writetable(T, fn);
end
