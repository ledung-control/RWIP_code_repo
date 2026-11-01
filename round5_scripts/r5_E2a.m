function r5_E2a(stage)
% Round 5 E2a (protocol v1.7, written before running): retrain BC-only and BC+TD3 with a bias-free actor so that pi(0) = 0 EXACTLY (architectural zero at the equilibrium
% observation).  Everything else = the paper configuration (gamma 0.995, S_PH 10, same schedule, authors' environment/reward/critic).  NEW IMPLEMENTATION: Train_RL_v13_bc_E2a.m (copy, 3 edits).
% stage 'smoke': seed 101, tiny schedule, only to check that pi(0) stays 0 after BC and after TD3 and that saving/evaluation work.  Not a result.
% stage 'final': seeds 1..5, SkipBCGate = true (labelled deviation: the authors' BC gate stopped 3 of 18 + 2 of 5 earlier runs).  Resumable.
% Evaluation: S0 60 s and 600 s, IC [0.10;0;0;0], same evaluator as E1; pi(0) printed for every agent.
if nargin<1, stage='final'; end
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(here); addpath(proj); addpath(here);
tag = ['MATLAB_' strrep(version('-release'),' ','')]; sub = ['E2a_' stage];
Rdir = fullfile(here,'..','results',sub); Ldir = fullfile(here,'..','logs',sub); Adir = fullfile(here,'..','agents',sub);
for d = {Rdir,Ldir,Adir}, if ~isfolder(d{1}), mkdir(d{1}); end, end
if strcmp(stage,'smoke'), seeds = 101; smoke = true; else, seeds = 1:5; smoke = false; end
for seed = seeds
  id = sprintf('E2a_s%d',seed); sd = fullfile(Adir,id); outcsv = fullfile(Rdir,sprintf('%s_%s.csv',id,tag));
  if isfile(outcsv), fprintf('skip %s\n',id); continue; end
  if isfolder(sd) && numel(dir(sd))>2, movefile(sd,[sd '_partial_' datestr(now,'yyyymmdd_HHMMSS')]); end
  diary(fullfile(Ldir,sprintf('%s_%s_%s.txt',id,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
  fprintf('r5_E2a[%s] %s %s %s\n', stage, id, datestr(now,31), version);
  t0 = tic; ok = true;
  try, r5_E2a_train(seed, sd, true, smoke);
  catch ME, ok = false; fprintf(2,'TRAINING ERROR: %s\n', ME.message); for k=1:min(6,numel(ME.stack)), fprintf(2,'  at %s line %d\n', ME.stack(k).name, ME.stack(k).line); end, end
  wall = toc(t0); fprintf('training wall %.1f s ok=%d\n', wall, ok);
  d = dir(fullfile(sd,'*.mat')); for i=1:numel(d), fprintf('  produced %s\n', d(i).name); end
  if ok
    try
      P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; oEq = rl_obs_v13(0,0,0,0,0,0); rows = {};
      pairs = {fullfile(sd,sprintf('agent_bconly_seed%d.mat',seed)),'BC-only'; fullfile(sd,sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3'};
      for i=1:size(pairs,1)
        if ~isfile(pairs{i,1}), fprintf('missing %s\n', pairs{i,1}); continue; end
        S = load(pairs{i,1}); ag = S.agent; a0 = pol(ag,oEq); fprintf('%s pi(0) = %.3e\n', pairs{i,2}, a0);
        for Tf = [60 600]
          P = P0; P.rlPolicy = @(o) pol(ag,o); P.rlTs = S.cfg.Ts_agent; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax;
          o = sim_rwip_v13('PSOFLC_RL', P, IC, Tf, 1e-3);
          rows(end+1,:) = {id,seed,pairs{i,2},Tf,a0,o.IAEtheta,o.IAEphi,o.OS_theta,o.ts_theta,o.phi_end,o.rho,wall}; %#ok<AGROW>
          fprintf('%-7s T=%3d IAEth=%.7g IAEphi=%.6g phi_end=%.4g\n', pairs{i,2}, Tf, o.IAEtheta, o.IAEphi, o.phi_end);
        end
      end
      T = cell2table(rows,'VariableNames',{'RunID','Seed','Group','Tf','pi0','IAEtheta','IAEphi','OS_theta','ts_theta','phi_end','rho','wall_s'}); writetable(T,outcsv);
    catch ME, fprintf(2,'EVAL ERROR: %s\n', ME.message); end
  end
  fprintf('DONE %s %s\n', id, datestr(now,31)); diary off
end
end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
