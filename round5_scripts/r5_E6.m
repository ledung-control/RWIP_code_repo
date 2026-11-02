function r5_E6(stage, nEp, seeds)
% Round 5 E6 (R2-5): end-to-end TD3 baseline, trained from scratch (no behaviour cloning, no FLC / SMC-ESO / cascade), authors' plant and evaluator.
% stage: 'smoke' (plant validation + 4 training episodes) | 'train' (nEp episodes per seed) | 'eval' (all agents in the S1/S2 cases of E3).
% Protocol v2.1 (written before the runs): see Round5_Protocol_E1-E8.md.
if nargin<1, stage='smoke'; end, if nargin<2, nEp=800; end, if nargin<3, seeds=1:3; end
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(here); addpath(proj);
tag = ['MATLAB_' strrep(version('-release'),' ','')];
Rdir=fullfile(here,'..','results',['E6_' stage]); Ldir=fullfile(here,'..','logs',['E6_' stage]); Adir=fullfile(here,'..','agents','E6');
for d={Rdir,Ldir,Adir}, if ~isfolder(d{1}), mkdir(d{1}); end, end
diary(fullfile(Ldir,sprintf('E6_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E6[%s] %s %s nEp=%d\n',stage,datestr(now,31),version,nEp);
[env,cfg] = r5_E6_env();
switch stage
  case 'smoke'
    validatePlant(cfg);
    trainSeed(env,cfg,99,4,Adir,true);
  case 'train'
    for s=seeds, trainSeed(env,cfg,s,nEp,Adir,false); end
  case 'eval'
    evalAll(cfg,Adir,Rdir,tag,seeds);
end
fprintf('DONE %s\n',datestr(now,31)); diary off
end

function trainSeed(env,cfg,seed,nEp,Adir,smoke)
fin = fullfile(Adir,sprintf('e2e_seed%d.mat',seed));
if ~smoke && isfile(fin), fprintf('skip seed %d\n',seed); return; end
rng(seed,'twister');
oi=getObservationInfo(env); ai=getActionInfo(env);
agent = rlTD3Agent(oi,ai,rlAgentInitializationOptions('NumHiddenUnit',256));
o = agent.AgentOptions; o.SampleTime=cfg.Ts_agent; o.DiscountFactor=0.99; o.MiniBatchSize=256; o.ExperienceBufferLength=1e6; o.TargetSmoothFactor=5e-3; o.PolicyUpdateFrequency=2;
o.ActorOptimizerOptions.LearnRate=1e-4; for k=1:numel(o.CriticOptimizerOptions), o.CriticOptimizerOptions(k).LearnRate=1e-3; end
o.ExplorationModel.StandardDeviation=0.2; o.ExplorationModel.StandardDeviationDecayRate=0; o.ExplorationModel.StandardDeviationMin=0.2;
agent.AgentOptions=o;
sd = fullfile(Adir,sprintf('ckpt_seed%d',seed)); if ~isfolder(sd), mkdir(sd); end
to = rlTrainingOptions('MaxEpisodes',nEp,'MaxStepsPerEpisode',round(cfg.Tf/cfg.Ts_agent),'StopTrainingCriteria','none','Plots','none','Verbose',true, ...
    'ScoreAveragingWindowLength',20,'SaveAgentCriteria','EpisodeFrequency','SaveAgentValue',100,'SaveAgentDirectory',sd);
t0=tic; stats = train(agent,env,to); el=toc(t0);
fprintf('seed %d trained %d episodes in %.1f min\n',seed,nEp,el/60);
if ~smoke, save(fin,'agent','stats','seed','nEp','cfg'); else, save(fullfile(Adir,'e2e_smoke.mat'),'agent'); end
% quick check: nominal case
M = rollout(agent,cfg.P,[0.1;0;0;0],10,NaN); fprintf('seed %d quick nominal 10 s: IAEth=%.4g IAEphi=%.4g diverged=%d\n',seed,M.IAEth,M.IAEphi,M.div);
end

function M = rollout(agent,P,x0,Tf,kick)
Ts=1e-3; nA=round(Tf/Ts); x=x0; V=0; kicked=isnan(kick); th=zeros(nA+1,1); ph=zeros(nA+1,1); th(1)=x(1); ph(1)=x(3); div=false;
actor = []; if ~isempty(agent), actor=getActor(agent); end
for k=1:nA
  if mod(k-1,10)==0
    if isempty(agent), V=0; elseif isa(agent,'function_handle'), V=agent(x,P); else
      a=getAction(actor,{[x(1)/0.15; x(2)/3; x(3)/10; x(4)/20]}); V=24*max(-1,min(1,double(a{1}))); end
  end
  x = rk4(x,V,P,Ts);
  if ~kicked && k*Ts>=5.0, x(2)=x(2)+kick; kicked=true; end
  th(k+1)=x(1); ph(k+1)=x(3);
  if ~all(isfinite(x)) || abs(x(1))>1.2, div=true; th(k+2:end)=NaN; ph(k+2:end)=NaN; break; end
end
t=(0:nA)'*Ts; M.div=div;
if div, M.IAEth=NaN; M.IAEphi=NaN; M.rec=false; M.thEnd=NaN; M.phEnd=NaN; return; end
M.IAEth=trapz(t,abs(th)); M.IAEphi=trapz(t,abs(ph)); w=t>t(end)-10; M.thEnd=max(abs(th(w))); M.phEnd=max(abs(ph(w)));
M.rec = M.thEnd<0.01 && M.phEnd<0.05;
end

function evalAll(cfg,Adir,Rdir,tag,seeds)
P0=cfg.P;
pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
rows={};
for s=seeds
  f=fullfile(Adir,sprintf('e2e_seed%d.mat',s)); if ~isfile(f), continue; end
  S=load(f); ag=S.agent;
  cases={}; for ip=1:size(pert,1), cases(end+1,:)={'S1',pert{ip,1},NaN,ip}; end, for k=[1 2], cases(end+1,:)={'S2','kick',k,0}; end
  for ic=1:size(cases,1)
    P=P0; if strcmp(cases{ic,1},'S1'), ip=cases{ic,4}; for q=1:numel(pert{ip,2}), P.(pert{ip,2}{q})=P0.(pert{ip,2}{q})*pert{ip,3}{q}; end, kick=NaN; else, kick=cases{ic,3}; end
    M=rollout(ag,P,[0.1;0;0;0],60,kick);
    rows(end+1,:)={sprintf('E2E TD3 s%d',s),cases{ic,1},cases{ic,2},cases{ic,3},M.div,M.IAEth,M.IAEphi,M.thEnd,M.phEnd,M.rec}; %#ok<AGROW>
    fprintf('s%d %s %-9s k=%g div=%d IAEth=%.4g IAEphi=%.4g rec=%d\n',s,cases{ic,1},cases{ic,2},cases{ic,3},M.div,M.IAEth,M.IAEphi,M.rec);
  end
end
T=cell2table(rows,'VariableNames',{'Controller','Scenario','Case','Kick','Diverged','IAEtheta','IAEphi','thEnd','phEnd','Recovered'});
writetable(T,fullfile(Rdir,['E6_eval_' tag '.csv']));
end

function validatePlant(cfg)
% check of the copied plant + integrator: PSO-FLC through this simulator vs the stored evaluator value (IAEtheta 0.0244629, IAEphi 3.21104 at 1 kHz, nominal, 60 s)
P=cfg.P; P.g_theta_nom=-(P.Kt/P.Ra)/P.Ip_eq; fis=readfis(P.fisFile);
GRD=linspace(-1,1,201); [A,B]=ndgrid(GRD,GRD); SURF=reshape(evalfis(fis,[A(:) B(:)]),201,201);
Ts=1e-3; x=[0.1;0;0;0]; nA=60000; th=zeros(nA+1,1); ph=th; th(1)=x(1);
for k=1:nA
  tref=max(-P.theta_ref_max,min(P.theta_ref_max,-(P.Kp_phi_mag*x(3)+P.Kv_phi_mag*x(4))));
  i1=max(-1,min(1,P.Ke*(x(1)-tref))); i2=max(-1,min(1,P.Kde*x(2)));
  V=P.Ku*bil(GRD,SURF,i1,i2)+P.Kd_theta*x(2)+P.Kd_phi*x(4); V=max(-24,min(24,V));
  x=rk4(x,V,P,Ts); th(k+1)=x(1); ph(k+1)=x(3);
end
t=(0:nA)'*Ts; fprintf('validate: IAEtheta=%.7g (stored 0.0244629)  IAEphi=%.6g (stored 3.21104)\n',trapz(t,abs(th)),trapz(t,abs(ph)));
end
function v=bil(g,S,a,b)
n=numel(g); h=2/(n-1); xa=(max(-1,min(1,a))+1)/h; xb=(max(-1,min(1,b))+1)/h; i=min(max(floor(xa),0),n-2); j=min(max(floor(xb),0),n-2); fa=xa-i; fb=xb-j; i=i+1; j=j+1;
v=S(i,j)*(1-fa)*(1-fb)+S(i+1,j)*fa*(1-fb)+S(i,j+1)*(1-fa)*fb+S(i+1,j+1)*fa*fb;
end
function x=rk4(x,V,P,Ts)
f=@(X) pd(X,V,P); k1=f(x); k2=f(x+Ts/2*k1); k3=f(x+Ts/2*k2); k4=f(x+Ts*k3); x=x+Ts/6*(k1+2*k2+2*k3+k4);
end
function dx=pd(x,V,P)
th=x(1); thd=x(2); phd=x(4); tau=(P.Kt/P.Ra)*(V-P.Ke_m*phd); thdd=(-tau+P.br*phd+P.mgl_eq*sin(th)-P.bp*thd)/P.Ip_eq; dx=[thd; thdd; phd; (tau-P.br*phd)/P.Ir-thdd];
end
