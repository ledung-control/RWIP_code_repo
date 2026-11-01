function r5_E3b(stage)
% Round 5 E3b (protocol v2.0, written before running): PSO-FLC re-optimized with the paper's PSO (30x20, same cost/specs/bounds/init) under 100 Hz outer-loop + fuzzy sampling.
% Evaluator = sim_rwip_r5 (RK4 1 ms plant, ctrlTs = 1e-2 ZOH), FIS rebuilt as in buildFIS of pso_rwip_v13.m.  stage 'smoke' | 'final'.
if nargin<1, stage='smoke'; end
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(here); addpath(proj);
tag = ['MATLAB_' strrep(version('-release'),' ','')];
Rdir = fullfile(here,'..','results',['E3b_' stage]); Ldir = fullfile(here,'..','logs',['E3b_' stage]);
for d={Rdir,Ldir}, if ~isfolder(d{1}), mkdir(d{1}); end, end
diary(fullfile(Ldir,sprintf('E3b_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E3b[%s] %s %s\n', stage, datestr(now,31), version);
ctx = struct(); ctx.P0 = make_P_v13('fis_rwip_v13_best.fis'); ctx.P0.g_theta_nom = -(ctx.P0.Kt/ctx.P0.Ra)/ctx.P0.Ip_eq;
ctx.Tf=30; ctx.Ts=1e-3; ctx.IC=[0.10;0;0;0]; ctx.Tos=4; ctx.TendWin=8; ctx.thetaTol=0.002; ctx.phiTol=0.020; ctx.thetaHardLimit=1.2; ctx.thetaExceedPenalty=1e5; ctx.J_simError=1e14;
ctx.spec_OS=0.006; ctx.spec_tsTheta=3.0; ctx.spec_phiMax=0.52; ctx.spec_tsPhi=11.0; ctx.wCon=1e3; ctx.wCon2=1e4;
ctx.ru=0.002; ctx.rdu=2e-4; ctx.rtvdu=1e-4; ctx.wEnd_theta=400; ctx.wEnd_phi=60; ctx.wSlope_phi=400; ctx.wPhiDotEnd=200; ctx.rGain=1e-4;
VarMin=[0.15 0.15 0.15 0.5 0.05 4.0 0.0 -4.0 0.000 0.000]; VarMax=[0.95 0.95 0.95 6.0 1.00 24.0 4.0 0.4 0.030 0.120]; nVar=10;
xOff = [0.20969 0.78364 0.290554 1.93975 0.678232 23.4172 2.91123 0.190429 0.00437824 0.0183369];   % official solution (Table 2)
if strcmp(stage,'smoke')
  for rate = [1e-3 1e-2]
    ctx.ctrlTs = rate; t0=tic; [J,M] = evalCostR5(xOff,ctx); w=toc(t0);
    fprintf('official x, ctrlTs=%g s: J=%.6g gv=%.4g OS=%.5g ts_th=%.3f phi_pk=%.4f ts_phi=%.3f  (%.1f s per evaluation)\n', rate,J,M.gv,M.OS_theta,M.ts_theta,M.phi_peak,M.ts_phi,w);
  end
  fprintf('stored paper J = 0.314771 (OS 0.0059865, ts_th 2.78, phi_pk 0.5194, ts_phi 10.72)\n');
  ctx.ctrlTs=1e-2; x2=xOff; x2(1:3)=0.9; [J2,~]=evalCostR5(x2,ctx); fprintf('FIS acts: J(s=0.9)=%.6g\n',J2);
  diary off; return
end
% ---------------- final: five PSO runs ----------------
ctx.ctrlTs = 1e-2; nPop=30; MaxIt=20; w0=0.9; wdamp=0.97; c1=1.5; c2=1.7;
seeds = 20260806 + (0:4);
hist = fullfile(Rdir,['E3b_pso_' tag '.csv']);
for ks = 1:numel(seeds)
  outk = fullfile(Rdir,sprintf('E3b_pso_run%d_%s.mat',ks,tag)); if isfile(outk), fprintf('skip run %d\n',ks); continue; end
  rng(seeds(ks),'twister'); w=w0;
  Pos = VarMin + rand(nPop,nVar).*(VarMax-VarMin); Vel = zeros(nPop,nVar);
  Pos(1,:) = [0.2000 0.77182 0.31824 1.8823 0.66601 24.0 2.6788 0.19785 0.0043169 0.0184026];
  Pos(2,:) = [0.5000 0.50000 0.5000 2.0000 0.33330 24.0 3.0000 0.00 0.011000 0.047000];
  Pos(3,:) = [0.2000 0.77182 0.31824 2.0030 0.69100 23.10 2.6060 0.15800 0.0044000 0.0188000];
  nLocal = max(4, round(0.75*nPop)); base = Pos(1,:); sig = max(0.08*abs(base), 0.02*(VarMax-VarMin).*(abs(base)<eps));
  for i=4:nLocal, Pos(i,:) = min(max(base + sig.*randn(1,nVar), VarMin), VarMax); end
  VelMax = max(0.25*abs(base), 0.01*(VarMax-VarMin)); VelMin = -VelMax;
  Costs = inf(nPop,1); for i=1:nPop, Costs(i) = evalCostR5(Pos(i,:),ctx); end
  PB=Pos; PBc=Costs; [Gc,gi]=min(Costs); G=Pos(gi,:); BH=zeros(MaxIt,1);
  for it=1:MaxIt
    for i=1:nPop
      r1=rand(1,nVar); r2=rand(1,nVar);
      Vel(i,:) = w*Vel(i,:) + c1*r1.*(PB(i,:)-Pos(i,:)) + c2*r2.*(G-Pos(i,:));
      Vel(i,:) = min(max(Vel(i,:),VelMin),VelMax); Pos(i,:) = Pos(i,:)+Vel(i,:);
      lo=Pos(i,:)<VarMin; hi=Pos(i,:)>VarMax; Pos(i,lo)=VarMin(lo); Pos(i,hi)=VarMax(hi); Vel(i,lo|hi) = -0.5*Vel(i,lo|hi);
    end
    for i=1:nPop, Costs(i)=evalCostR5(Pos(i,:),ctx); end
    imp = Costs<PBc; PB(imp,:)=Pos(imp,:); PBc(imp)=Costs(imp); [mc,mi]=min(PBc); if mc<Gc, Gc=mc; G=PB(mi,:); end
    BH(it)=Gc; w=w*wdamp; fprintf('run %d iter %2d/%d J=%.6g\n',ks,it,MaxIt,Gc);
  end
  [J,M] = evalCostR5(G,ctx); save(outk,'G','Gc','BH','M','seeds','ks');
  fprintf('run %d final J=%.6g gv=%.4g OS=%.5g ts_th=%.3f phi_pk=%.4f ts_phi=%.3f x=%s\n',ks,J,M.gv,M.OS_theta,M.ts_theta,M.phi_peak,M.ts_phi,mat2str(G,6));
  row = table(ks,seeds(ks),J,M.gv,M.OS_theta,M.ts_theta,M.phi_peak,M.ts_phi,string(mat2str(G,8)),'VariableNames',{'Run','RngSeed','J','gv','OS_theta','ts_theta','phi_peak','ts_phi','x'});
  writetable(row,hist,'WriteMode','append','WriteVariableNames',~isfile(hist));
end
% ---------------- evaluation at 100 Hz in the E3 scenarios ----------------
evalcsv = fullfile(Rdir,['E3b_eval_' tag '.csv']);
if ~isfile(evalcsv)
  variants = {'original', xOff}; for ks=1:5, S=load(fullfile(Rdir,sprintf('E3b_pso_run%d_%s.mat',ks,tag))); variants(end+1,:) = {sprintf('retuned run%d',ks), S.G}; end %#ok<AGROW>
  pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
  rec = @(o) ~o.diverged && isfinite(o.IAEtheta) && max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && max(abs(o.phi(o.t > o.t(end)-10))) < 0.05;
  rows = {};
  for iv=1:size(variants,1)
    x = variants{iv,2}; P0 = paramsFromX(x,ctx.P0);
    cases = {}; for ip=1:size(pert,1), cases(end+1,:) = {'S1',pert{ip,1},NaN,ip}; end
    for k=[1.0 2.0], cases(end+1,:) = {'S2','kick',k,0}; end
    for ic=1:size(cases,1)
      P = P0; P.ctrlTs=1e-2; P.smcTs=1e-3; P.rlTs=1e-2;
      if strcmp(cases{ic,1},'S1'), ip=cases{ic,4}; for f=1:numel(pert{ip,2}), P.(pert{ip,2}{f}) = ctx.P0.(pert{ip,2}{f})*pert{ip,3}{f}; end
      else, P.dist = struct('t',5.0,'dthetadot',cases{ic,3}); end
      o = sim_rwip_r5('PSOFLC',P,[0.10;0;0;0],60,1e-3); dv = o.diverged;
      rows(end+1,:) = {variants{iv,1},cases{ic,1},cases{ic,2},cases{ic,3},dv,o.IAEtheta,o.IAEphi,o.OS_theta,o.ts_theta,o.ts_phi,o.Urms,o.phi_end,rec(o)}; %#ok<AGROW>
      fprintf('%-14s %s %-9s k=%g IAEth=%.5g IAEphi=%.5g ts_phi=%.3f rec=%d\n',variants{iv,1},cases{ic,1},cases{ic,2},cases{ic,3},o.IAEtheta,o.IAEphi,o.ts_phi,rec(o));
    end
  end
  T = cell2table(rows,'VariableNames',{'Variant','Scenario','Case','Kick','Diverged','IAEtheta','IAEphi','OS_theta','ts_theta','ts_phi','Urms','phi_end','Recovered'}); writetable(T,evalcsv);
end
fprintf('DONE %s\n',datestr(now,31)); diary off
end
function P = paramsFromX(x,P0)
P = P0; fn = [tempname '.fis']; fis = buildFIS(x(1),x(2),x(3)); writeFIS(fis,fn); P.fisFile = fn;
P.Ke=x(4); P.Kde=x(5); P.Ku=x(6); P.Kd_theta=x(7); P.Kd_phi=x(8); P.Kp_phi_mag=x(9); P.Kv_phi_mag=x(10);
end
function [J,M] = evalCostR5(x,ctx)
M = struct('OS_theta',NaN,'ts_theta',NaN,'phi_peak',NaN,'ts_phi',NaN,'gv',NaN); J = ctx.J_simError;
try
  P = paramsFromX(x,ctx.P0); P.ctrlTs = ctx.ctrlTs; P.smcTs = ctx.Ts; P.rlTs = 1e-2;
  o = sim_rwip_r5('PSOFLC',P,ctx.IC,ctx.Tf,ctx.Ts); try, delete(P.fisFile); catch, end
  if o.diverged, return; end
  t=o.t(:); theta=o.theta(:); phi=o.phi(:); u=o.u(:); phid=o.phi_dot(:); dt = t(2)-t(1);
  e_theta=theta; e_phi=phi;
  idxOS = t <= ctx.Tos;
  if theta(1) >= 0, OS = max(0, 0 - min(theta(idxOS))); else, OS = max(0, max(theta(idxOS))); end
  Eu=trapz(t,u.^2); du=[0;diff(u)]/dt; Edu=trapz(t,du.^2); tv_du=sum(abs(diff(du)));
  idxEnd = t >= max(t(1), t(end)-ctx.TendWin);
  theta_end=theta(end); phi_end=phi(end); phid_end=mean(abs(phid(idxEnd)));
  tt=t(idxEnd); pp=phi(idxEnd); A=[tt ones(numel(tt),1)]; cf=A\pp; phi_slope=cf(1);
  ts_theta = lastAbove(t,e_theta,ctx.thetaTol); ts_phi = lastAbove(t,e_phi,ctx.phiTol); phi_peak = max(abs(e_phi));
  gv = max(0,OS/ctx.spec_OS-1) + max(0,ts_theta/ctx.spec_tsTheta-1) + max(0,phi_peak/ctx.spec_phiMax-1) + max(0,ts_phi/ctx.spec_tsPhi-1);
  Jobj = ctx.ru*Eu + ctx.rdu*Edu + ctx.rtvdu*tv_du + ctx.wEnd_theta*(theta_end^2) + ctx.wEnd_phi*(phi_end^2) + ctx.wSlope_phi*(phi_slope^2) + ctx.wPhiDotEnd*(phid_end^2) + ctx.rGain*(x(7)^2+x(8)^2);
  J = Jobj + ctx.wCon*gv + ctx.wCon2*gv^2;
  if max(abs(theta)) > ctx.thetaHardLimit, J = J + ctx.thetaExceedPenalty; end
  if ~isfinite(J), J = ctx.J_simError; end
  M.OS_theta=OS; M.ts_theta=ts_theta; M.phi_peak=phi_peak; M.ts_phi=ts_phi; M.gv=gv;
catch ME
  fprintf(2,'eval error: %s\n',ME.message); J = ctx.J_simError;
end
end
function ts = lastAbove(t,e,tol), k = find(abs(e)>tol,1,'last'); if isempty(k), ts=0; else, ts=t(k); end, end
function fis = buildFIS(s_e,s_de,s_u)
s_e=min(max(s_e,0.10),0.95); s_de=min(max(s_de,0.10),0.95); s_u=min(max(s_u,0.10),0.95);
fis = mamfis('Name','fis_rwip_sym','AndMethod','min','OrMethod','max','ImplicationMethod','min','AggregationMethod','max','DefuzzificationMethod','centroid');
fis = addInput(fis,[-1 1],'Name','e_theta'); fis = addInput(fis,[-1 1],'Name','de_theta'); fis = addOutput(fis,[-1 1],'Name','u');
fis = addSymMFs(fis,'e_theta',s_e); fis = addSymMFs(fis,'de_theta',s_de); fis = addSymMFs(fis,'u',s_u);
rl = zeros(25,4); r=0; for i=1:5, for j=1:5, r=r+1; rl(r,:) = [i,j,min(5,max(1,i+j-3)),1]; end, end
fis = addRule(fis,[rl ones(25,1)]);
end
function fis = addSymMFs(fis,varName,s)
nm = {'NB','NS','Z','PS','PB'}; P = {[-1 -1 -1 -s],[-1 -s 0],[-s 0 s],[0 s 1],[s 1 1 1]}; T = {'trapmf','trimf','trimf','trimf','trapmf'};
for k=1:5, fis = addMF(fis,varName,T{k},P{k},'Name',nm{k}); end
end
