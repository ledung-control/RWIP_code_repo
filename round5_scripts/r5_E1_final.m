function r5_E1_final(stage, budgetMin)
% Round 5, protocol v1.3 (dated 27 Oct 2026, written before these runs).  NEW IMPLEMENTATION wrapper; authors' code unchanged.
% stage 'bcgate' : re-run the three seed-102 tuning runs that stopped at the authors' BC gate (c4,c5,c6) with SkipBCGate=true.
%                  LABELLED PROTOCOL DEVIATION.  The original gate stops stay in the record (logs\E1, no CSV); outputs go to
%                  agents\E1_bcgate, results\E1_bcgate, logs\E1_bcgate.  These runs are NOT merged into the selection of v1.0/v1.2.
% stage 'final'  : config c3 (S_PH 10, gamma 0.9995) x seeds 1..5 (n=5, fixed in v1.3 before running).  For every seed it
%                  evaluates (a) the new BC-only and BC+TD3 agents and (b) the stored paper agents (rl_v13_repro) with the same
%                  evaluator, raw and pi(0)-removed, 60 s and 600 s, IC [0.10;0;0;0].  Outputs: agents\E1_final, results\E1_final, logs\E1_final.
% Resume: a run is skipped when its CSV exists; an interrupted folder is moved aside (never overwritten).  File 'STOP' in this
% folder prevents new starts.  budgetMin = minutes after which no NEW run starts (Inf = continuous).
if nargin<1, stage='final'; end
if nargin<2, budgetMin = Inf; end
tSession = tic;
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(here); addpath(proj);
tag = ['MATLAB_' strrep(version('-release'),' ','')];
shadow = fullfile(here,'E1_shadow_SPH40'); CF = [10 0.995; 10 0.999; 10 0.9995; 40 0.995; 40 0.999; 40 0.9995];
switch stage
  case 'bcgate', jobs = [4 102; 5 102; 6 102]; skipBC = true;  sub = 'E1_bcgate'; Tevals = 60;
  case 'final',  jobs = [3 1; 3 2; 3 3; 3 4; 3 5];            skipBC = false; sub = 'E1_final';  Tevals = [60 600];
  case 'final_skipbc', jobs = [3 3; 3 5];                  skipBC = true;  sub = 'E1_final_skipbc'; Tevals = [60 600];  % v1.4: seeds 3,5 stopped at BC gate in 'final'; labelled deviation
  otherwise, error('stage must be bcgate, final or final_skipbc');
end
Rdir = fullfile(here,'..','results',sub); Ldir = fullfile(here,'..','logs',sub); Adir = fullfile(here,'..','agents',sub);
for d = {Rdir,Ldir,Adir}, if ~isfolder(d{1}), mkdir(d{1}); end, end
src = fileread(fullfile(proj,'rl_obs_v13.m'));
if ~isfolder(shadow), mkdir(shadow); end
fid = fopen(fullfile(shadow,'rl_obs_v13.m'),'w'); fwrite(fid,regexprep(src,'S_PH\s*=\s*10\.0;','S_PH   = 40.0;  % R5-E1 SHADOW COPY','once')); fclose(fid);
for j = 1:size(jobs,1)
  c = jobs(j,1); seed = jobs(j,2); SPH = CF(c,1); gam = CF(c,2); id = sprintf('c%d_SPH%d_g%g_s%d',c,SPH,gam,seed);
  sd = fullfile(Adir,id); outcsv = fullfile(Rdir,sprintf('%s_%s_%s.csv',sub,id,tag));
  if isfile(outcsv), fprintf('skip %s (done)\n', id); continue; end
  if toc(tSession)/60 > budgetMin || isfile(fullfile(here,'STOP')), fprintf('window reached or STOP present: not starting %s\n', id); break; end
  if isfolder(sd) && numel(dir(sd))>2, movefile(sd,[sd '_partial_' datestr(now,'yyyymmdd_HHMMSS')]); fprintf('moved partial folder of %s aside\n', id); end
  rmpath(shadow); if SPH~=10, addpath(shadow,'-begin'); end
  clear functions; rehash;
  t = rl_obs_v13(0,0,10,0,0,0); assert(abs(t(3)-10/SPH)<1e-12,'observation scaling not as intended');
  diary(fullfile(Ldir,sprintf('%s_%s_%s_%s.txt',sub,id,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
  fprintf('r5_E1_final[%s] %s  %s  %s\nS_PH=%g gamma=%g seed=%d SkipBCGate=%d | rl_obs_v13 = %s\nSaveDir %s\n', stage, id, datestr(now,31), version, SPH, gam, seed, skipBC, which('rl_obs_v13'), sd);
  t0 = tic; ok = true;
  try, r5_E1_train2(seed, gam, sd, skipBC);
  catch ME, ok = false; fprintf(2,'TRAINING ERROR: %s\n', ME.message); for k=1:min(6,numel(ME.stack)), fprintf(2,'  at %s line %d\n', ME.stack(k).name, ME.stack(k).line); end, end
  wall = toc(t0); fprintf('training wall time %.1f s (ok=%d)\n', wall, ok);
  d = dir(fullfile(sd,'*.mat')); for i=1:numel(d), fprintf('  produced %s (%d bytes)\n', d(i).name, d(i).bytes); end
  if ok
    try
      P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; oEq = rl_obs_v13(0,0,0,0,0,0); rows = {};
      pairs = {fullfile(sd,sprintf('agent_bconly_seed%d.mat',seed)),'BC-only','new'; fullfile(sd,sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3','new'};
      if startsWith(stage,'final')   % stored paper agents, same evaluator (default S_PH 10 path)
        pairs(end+1,:) = {fullfile(proj,'rl_v13_repro',sprintf('agent_bconly_seed%d.mat',seed)),'BC-only','paper_stored'};
        pairs(end+1,:) = {fullfile(proj,'rl_v13_repro',sprintf('agent_bc_seed%d.mat',seed)),'BC+TD3','paper_stored'};
      end
      for i=1:size(pairs,1)
        if ~isfile(pairs{i,1}), fprintf('missing %s\n', pairs{i,1}); continue; end
        S = load(pairs{i,1}); ag = S.agent; a0 = pol(ag,oEq);
        for Tf = Tevals
          for var = {'raw','unbiased'}
            ab = 0; if strcmp(var{1},'unbiased'), ab = a0; end
            P = P0; P.rlPolicy = @(o) pol(ag,o)-ab; P.rlTs = S.cfg.Ts_agent; P.rlUmax = S.cfg.uMax; P.uCorrMax = S.cfg.uMax;
            o = sim_rwip_v13('PSOFLC_RL', P, IC, Tf, 1e-3);
            rows(end+1,:) = {id,c,SPH,gam,seed,pairs{i,3},pairs{i,2},var{1},Tf,a0,o.IAEtheta,o.IAEphi,o.OS_theta,o.ts_theta,o.phi_end,o.rho,wall}; %#ok<AGROW>
            fprintf('%-12s %-7s %-8s T=%3d pi(0)=%+.6f IAEth=%.7g IAEphi=%.6g phi_end=%.4g\n', pairs{i,3}, pairs{i,2}, var{1}, Tf, a0, o.IAEtheta, o.IAEphi, o.phi_end);
          end
        end
      end
      T = cell2table(rows,'VariableNames',{'RunID','Config','S_PH','gamma','Seed','Source','Group','Variant','Tf','pi0','IAEtheta','IAEphi','OS_theta','ts_theta','phi_end','rho','wall_s'});
      writetable(T, outcsv);
    catch ME, fprintf(2,'EVAL ERROR: %s\n', ME.message); end
  end
  fprintf('DONE %s %s\n', id, datestr(now,31)); diary off
end
rmpath(shadow); clear functions;
end
function a = pol(ag,o), a = getAction(ag,{o(:)}); if iscell(a), a=a{1}; end; a = double(a(1)); end
