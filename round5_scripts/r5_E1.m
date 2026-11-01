function r5_E1(stage, budgetMin)
% E1 (protocol v1.0 section 4): retrain with corrected phi-observation scaling (S_PH) and discount factor (gamma).
% NEW IMPLEMENTATION (labelled as such): (i) a shadow copy of rl_obs_v13.m with another S_PH, put first on the path
% (the authors' file is not modified; the shadow is used by BOTH training env and evaluator), (ii) r5_E1_train.m = copy of the
% call in Train_RL_v13_repro.m with DiscountFactor as a parameter.  Everything else is the authors' code, unchanged.
% stage: 'pilot'  -> config c4 (S_PH 40, gamma 0.995), tuning seed 101 (one training)
%        'tuning' -> all 6 configs x seeds 101..103 (18 trainings); runs already done are skipped, nothing is overwritten.
% Configs: c1 (10,0.995)=paper  c2 (10,0.999)  c3 (10,0.9995)  c4 (40,0.995)  c5 (40,0.999)  c6 (40,0.9995)
if nargin<1, stage='pilot'; end
if nargin<2, budgetMin = Inf; end   % session window: no NEW training is started after budgetMin minutes (a started run always finishes; runs cannot be paused)
tSession = tic;
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(here); addpath(proj);  % NOT cd(proj): the current folder would shadow the E1 shadow copy of rl_obs_v13
Rdir = fullfile(here,'..','results','E1'); Ldir = fullfile(here,'..','logs','E1'); Adir = fullfile(here,'..','agents','E1');
tag = ['MATLAB_' strrep(version('-release'),' ','')];
CF = [10 0.995; 10 0.999; 10 0.9995; 40 0.995; 40 0.999; 40 0.9995];
switch stage
  case 'pilot',  jobs = [4 101];
  case 'tuning', jobs = []; for s=101:103, for c=1:6, jobs(end+1,:) = [c s]; end, end %#ok<AGROW>
  otherwise, error('stage must be pilot or tuning');
end
shadow = fullfile(here,'E1_shadow_SPH40'); if ~isfolder(shadow), mkdir(shadow); end
src = fileread(fullfile(proj,'rl_obs_v13.m'));
assert(~isempty(regexp(src,'S_PH\s*=\s*10\.0;','once')),'S_PH line not found in rl_obs_v13.m');
new = regexprep(src,'S_PH\s*=\s*10\.0;','S_PH   = 40.0;  % R5-E1 SHADOW COPY (only this line differs from the authors'' file)','once');
fid = fopen(fullfile(shadow,'rl_obs_v13.m'),'w'); fwrite(fid,new); fclose(fid);
for j = 1:size(jobs,1)
  c = jobs(j,1); seed = jobs(j,2); SPH = CF(c,1); gam = CF(c,2); id = sprintf('c%d_SPH%d_g%g_s%d',c,SPH,gam,seed);
  sd = fullfile(Adir,id); outcsv = fullfile(Rdir,sprintf('E1_%s_%s.csv',id,tag));
  if isfile(outcsv), fprintf('skip %s (done)\n', id); continue; end
  if toc(tSession)/60 > budgetMin || isfile(fullfile(here,'STOP')), fprintf('session window reached or STOP file present: not starting %s; rerun r5_E1(''tuning'',%g) later\n', id, budgetMin); break; end
  if isfolder(sd) && ~isempty(dir(fullfile(sd,'*'))) && numel(dir(sd))>2   % interrupted earlier run: keep it, do not overwrite
    movefile(sd, [sd '_partial_' datestr(now,'yyyymmdd_HHMMSS')]); fprintf('moved partial folder of %s aside\n', id);
  end
  rmpath(shadow); if SPH~=10, addpath(shadow,'-begin'); end
  clear functions; rehash;
  t = rl_obs_v13(0,0,10,0,0,0); assert(abs(t(3)-10/SPH)<1e-12, 'observation scaling not as intended');
  diary(fullfile(Ldir,sprintf('E1_%s_%s_%s.txt',id,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
  fprintf('r5_E1 %s  %s  %s\nS_PH=%g gamma=%g seed=%d | rl_obs_v13 = %s\nSaveDir %s\n', id, datestr(now,31), version, SPH, gam, seed, which('rl_obs_v13'), sd);
  t0 = tic; ok = true;
  try
    r5_E1_train(seed, gam, sd);
  catch ME
    ok = false; fprintf(2,'TRAINING ERROR: %s\n', ME.message); for k=1:min(6,numel(ME.stack)), fprintf(2,'  at %s line %d\n', ME.stack(k).name, ME.stack(k).line); end
  end
  wall = toc(t0); fprintf('training wall time %.1f s (ok=%d)\n', wall, ok);
  d = dir(fullfile(sd,'*.mat')); for i=1:numel(d), fprintf('  produced %s (%d bytes)\n', d(i).name, d(i).bytes); end
  if ok
    try
      P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; oEq = rl_obs_v13(0,0,0,0,0,0); rows = {};
      pairs = {fullfile(sd,sprintf('agent_bconly_seed%d.mat',seed)),'BC-only'; fullfile(sd,sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3'};
      for i=1:size(pairs,1)
        S = load(pairs{i,1}); ag = S.agent; a0 = pol(ag,oEq);
        for var = {'raw','unbiased'}
          ab = 0; if strcmp(var{1},'unbiased'), ab = a0; end
          P = P0; P.rlPolicy = @(o) pol(ag,o)-ab; P.rlTs = S.cfg.Ts_agent; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax;
          o = sim_rwip_v13('PSOFLC_RL', P, IC, 60, 1e-3);
          rows(end+1,:) = {id,c,SPH,gam,seed,pairs{i,2},var{1},a0,o.IAEtheta,o.IAEphi,o.OS_theta,o.ts_theta,o.phi_end,o.rho,wall}; %#ok<AGROW>
          fprintf('%-7s %-8s pi(0)=%+.6f IAEth=%.7g IAEphi=%.6g phi_end=%.4g rho=%.3f\n', pairs{i,2}, var{1}, a0, o.IAEtheta, o.IAEphi, o.phi_end, o.rho);
        end
      end
      T = cell2table(rows,'VariableNames',{'RunID','Config','S_PH','gamma','Seed','Group','Variant','pi0','IAEtheta','IAEphi','OS_theta','ts_theta','phi_end','rho','wall_s'});
      writetable(T, outcsv);
    catch ME
      fprintf(2,'EVAL ERROR: %s\n', ME.message);
    end
  end
  fprintf('DONE %s %s\n', id, datestr(now,31)); diary off
end
rmpath(shadow); clear functions;
end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
