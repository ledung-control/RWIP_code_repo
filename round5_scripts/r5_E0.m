function r5_E0(seed)
% E0 reproducibility pilot (protocol v1.0 section 2): retrain the authors' fixed configuration with Train_RL_v13_repro (unchanged)
% into a NEW folder, then evaluate raw / pi(0)-removed at 60 s and compare with the stored agent of the same seed.
if nargin<1, seed=1; end
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(proj);
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
sd = fullfile(here,'..','agents',sprintf('E0_repro_seed%d',seed)); sd = fullfile(char(sd));
diary(fullfile(Ldir,sprintf('E0_seed%d_%s_%s.txt',seed,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E0 seed %d  %s  %s\nSaveDir %s\nTrain_RL_v13_repro.m date %s | Train_RL_v13_bc.m date %s | rwip_env_v13.m date %s\n', seed, datestr(now,31), version, sd, ...
    getfd('Train_RL_v13_repro.m'), getfd('Train_RL_v13_bc.m'), getfd('rwip_env_v13.m'));
t0 = tic; ok = true;
try
  Train_RL_v13_repro(seed, struct('SaveDir',sd,'Plots','none'));
catch ME
  ok = false; fprintf(2,'TRAINING ERROR: %s\n', ME.message); for k=1:min(6,numel(ME.stack)), fprintf(2,'  at %s line %d\n', ME.stack(k).name, ME.stack(k).line); end
end
fprintf('training wall time %.1f s (ok=%d)\n', toc(t0), ok);
d = dir(fullfile(sd,'*.mat')); for i=1:numel(d), fprintf('  produced %s (%d bytes)\n', d(i).name, d(i).bytes); end
if ok
  P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; oEq = rl_obs_v13(0,0,0,0,0,0); rows = {};
  pairs = {fullfile(sd,sprintf('agent_bconly_seed%d.mat',seed)), fullfile(proj,'rl_v13_repro',sprintf('agent_bconly_seed%d.mat',seed)), 'BC-only';
           fullfile(sd,sprintf('agent_bc_seed%d.mat',seed)),     fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed)),     'BC+TD3'};
  for i=1:size(pairs,1)
    for w = 1:2   % 1 = new retrain, 2 = stored
      f = pairs{i,w}; if ~isfile(f), fprintf('missing %s\n', f); continue; end
      S = load(f); ag = S.agent; a0 = getAction(ag,{oEq(:)}); if iscell(a0), a0=a0{1}; end; a0 = double(a0(1));
      for var = {'raw','unbiased'}
        ab = 0; if strcmp(var{1},'unbiased'), ab = a0; end
        P = P0; P.rlPolicy = @(o) pol(ag,o)-ab; P.rlTs = S.cfg.Ts_agent; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax;
        o = sim_rwip_v13('PSOFLC_RL', P, IC, 60, 1e-3);
        src = {'retrained_R5','stored'}; rows(end+1,:) = {pairs{i,3}, seed, src{w}, var{1}, a0, o.IAEtheta, o.IAEphi, o.phi_end, o.rho}; %#ok<AGROW>
        fprintf('%-7s %-12s %-8s pi(0)=%+.6f IAEth=%.7g IAEphi=%.6g phi_end=%.4g\n', pairs{i,3}, src{w}, var{1}, a0, o.IAEtheta, o.IAEphi, o.phi_end);
      end
    end
  end
  T = cell2table(rows,'VariableNames',{'Group','Seed','Source','Variant','pi0','IAEtheta','IAEphi','phi_end','rho'});
  writetable(T, fullfile(Rdir, sprintf('E0_seed%d_%s.csv',seed,tag)));
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
function d = getfd(f), x = dir(which(f)); if isempty(x), d='(missing)'; else, d = x(1).date; end, end
