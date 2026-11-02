function [env, cfg] = r5_E6_env(cfg)
% Round 5 E6: end-to-end TD3 baseline environment (no BC, no FLC / SMC-ESO / cascade).
% Same plant as the authors' rwip_env_v13 (plantDeriv copied unchanged, RK4 1 ms, agent ZOH 100 Hz), same observation scaling as rl_obs_v13
% (first four entries), same training-scenario distribution (IC / kick / +-20 % parameters). Action a in [-1,1] -> V = 24 a (full voltage range).
% Reward: r = -0.1 * ( |th|/0.010 + 0.054 |phi|/0.100 + 0.05 (V/24)^2 + 0.02 (dV/24)^2 ) per 10 ms step, failure (|th| > 0.5 rad) = -636 and terminal
% (the authors' 'state' reward with k_dense 0.1 and failPen for gamma 0.99), truncation at Tf = IsDone 2.
if nargin<1, cfg=struct(); end
d.Tf=20; d.Ts=1e-3; d.Ts_agent=1e-2; d.Vmax=24; d.thetaFail=0.5; d.failPen=636; d.k_dense=0.1;
d.th_s=0.010; d.ph_s=0.100; d.w_ph=0.054; d.w_u=0.05; d.w_du=0.02;
d.p_ic=0.40; d.p_kick=0.40; d.p_param=0.20; d.ic_theta=[-0.20 0.20]; d.ic_thetad=[-0.30 0.30]; d.ic_phid=[-2 2]; d.kick_t=[3 10]; d.kick_amp=[0.5 3.0]; d.param_rel=[0.8 1.2];
fn=fieldnames(d); for k=1:numel(fn), if ~isfield(cfg,fn{k}), cfg.(fn{k})=d.(fn{k}); end, end
cfg.P = make_P_v13('fis_rwip_v13_best.fis');
obsInfo = rlNumericSpec([4 1],'Name','rwip_obs');
actInfo = rlNumericSpec([1 1],'LowerLimit',-1,'UpperLimit',1,'Name','V_norm');
env = rlFunctionEnv(obsInfo, actInfo, @(a,ls) stepFcn(a,ls,cfg), @() resetFcn(cfg));
end

function [obs, ls] = resetFcn(cfg)
ls.P = cfg.P; ls.kickT = inf; ls.kickA = 0; r = rand();
if r < cfg.p_ic, ls.scen='ic';
elseif r < cfg.p_ic+cfg.p_kick, ls.scen='kick'; ls.kickT=urand(cfg.kick_t); ls.kickA=urand(cfg.kick_amp)*sign(rand-0.5);
else, ls.scen='param'; for f={'mgl_eq','Ip_eq','Kt'}, ls.P.(f{1}) = cfg.P.(f{1})*urand(cfg.param_rel); end, end
ls.x = [urand(cfg.ic_theta); urand(cfg.ic_thetad); 0; urand(cfg.ic_phid)];
ls.k=0; ls.t=0; ls.uPrev=0; ls.kicked=~isfinite(ls.kickT);
obs = obsOf(ls.x);
end

function [obs, reward, isdone, ls] = stepFcn(action, ls, cfg)
P=ls.P; Ts=cfg.Ts; nSub=round(cfg.Ts_agent/Ts);
V = cfg.Vmax*max(-1,min(1,action(1)));
for s=1:nSub
    ls.x = rk4(ls.x, V, P, Ts); ls.t = ls.t+Ts;
    if ~ls.kicked && ls.t >= ls.kickT, ls.x(2)=ls.x(2)+ls.kickA; ls.kicked=true; end
    if ~all(isfinite(ls.x)) || abs(ls.x(1))>cfg.thetaFail, break; end
end
ls.k = ls.k+1; th=ls.x(1); ph=ls.x(3);
if ~all(isfinite(ls.x)) || abs(th)>cfg.thetaFail
    reward=-cfg.failPen; isdone=1; obs=zeros(4,1); return
end
cA = abs(th)/cfg.th_s + cfg.w_ph*abs(ph)/cfg.ph_s + cfg.w_u*(V/cfg.Vmax)^2 + cfg.w_du*((V-ls.uPrev)/cfg.Vmax)^2;
reward = -cfg.k_dense*cA; ls.uPrev = V;
if ls.k >= round(cfg.Tf/cfg.Ts_agent), isdone=2; else, isdone=0; end
obs = obsOf(ls.x);
end

function o = obsOf(x), o = [x(1)/0.15; x(2)/3.0; x(3)/10.0; x(4)/20.0]; end
function x = rk4(x,V,P,Ts)
f=@(X) plantDeriv(X,V,P); k1=f(x); k2=f(x+Ts/2*k1); k3=f(x+Ts/2*k2); k4=f(x+Ts*k3); x = x+Ts/6*(k1+2*k2+2*k3+k4);
end
function dx = plantDeriv(x,V,P)
th=x(1); thd=x(2); phd=x(4);
tau=(P.Kt/P.Ra)*(V-P.Ke_m*phd);
thdd=(-tau+P.br*phd+P.mgl_eq*sin(th)-P.bp*thd)/P.Ip_eq;
dx=[thd; thdd; phd; (tau-P.br*phd)/P.Ir-thdd];
end
function v = urand(r), v=r(1)+(r(2)-r(1))*rand(); end
