function r5_E4(stage)
% Round 5 E4 (protocol v1.0 sec. 4, details fixed in v1.5 BEFORE running): LQR state-feedback baseline with the same outer loop.  NEW CONTROLLER = new implementation (sim_rwip_r5.m, mode 'LQR').
%  Inner loop  V = -K*[theta - theta_ref; theta_dot; phi_dot],  K = lqr(A,B,Q,R) on the linearization about the upright equilibrium of the 3-state subsystem [theta theta_dot phi_dot] (phi does not enter the dynamics; it is handled by the outer loop).
%  Search (stage 'search'): 600 Latin-hypercube samples (rng 20261029) of log10 q = [q_theta q_thetadot q_phidot] in [-1 5; -3 3; -4 2], R = 1 (only ratios matter), scored by the PSO cost J of pso_rwip_v13.m (constants copied below, FROZEN here; Tf = 30 s, IC [0.10;0;0;0], Ts = 1 ms, 1 kHz).
%  Two outer-loop variants, each with its own 600-sample search:  'FLCouter' (PSO-FLC outer gains) and 'SMCouter' (SMC-ESO's own outer gains)  -> separates the cascade-gain effect from the inner-loop law.
%  Stage 'compare': best K of each variant, plus FLC and SMC-ESO under BOTH outer-loop gain sets, scenarios S0/S1/S2, 1 kHz and 100 Hz (ZOH on outer loop + inner loop).
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(here); addpath(proj);
if nargin<1, stage='search'; end
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
diary(fullfile(Ldir,sprintf('E4_%s_%s_%s.txt',stage,tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E4[%s] %s %s\n', stage, datestr(now,31), version);
P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; Ts=1e-3;
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;
% ---- linearization (3 states) ----
Ip=P0.Ip_eq; Ir=P0.Ir; bp=P0.bp; br=P0.br; Ra=P0.Ra; Kem=P0.Ke_m; Kt=P0.Kt; mgl=P0.mgl_eq;
f3 = @(y,V) rhs3(y,V,Ip,Ir,bp,br,Ra,Kem,Kt,mgl);
h = 1e-6; A = zeros(3); B = zeros(3,1);
for j=1:3, e = zeros(3,1); e(j)=h; A(:,j) = (f3(e,0)-f3(-e,0))/(2*h); end
B = (f3(zeros(3,1),h)-f3(zeros(3,1),-h))/(2*h);
fprintf('A =\n'); disp(A); fprintf('B =\n'); disp(B); fprintf('rank ctrb = %d\n', rank(ctrb(A,B)));
variants = {'FLCouter',[P0.Kp_phi_mag P0.Kv_phi_mag]; 'SMCouter',[P0.SMC_Kp_phi_mag P0.SMC_Kv_phi_mag]};
switch stage
  case 'search'
    rng(20261029,'twister'); nS = 600; lo = [-1 -3 -4]; hi = [5 3 2];
    U = zeros(nS,3); for d=1:3, U(:,d) = (randperm(nS).' - rand(nS,1))/nS; end
    LQ = lo + U.*(hi-lo);
    for iv = 1:size(variants,1)
      out = fullfile(Rdir,sprintf('E4_search_%s_%s.csv',variants{iv,1},tag));
      if isfile(out), fprintf('skip %s (done)\n', variants{iv,1}); continue; end
      R = []; 
      for i = 1:nS
        q = 10.^LQ(i,:); [K,~,~] = lqr(A,B,diag(q),1);
        P = P0; P.Kp_phi_mag = variants{iv,2}(1); P.Kv_phi_mag = variants{iv,2}(2); P.lqrK = K(:).';
        [J,M] = costJ(P, IC, Ts);
        R(end+1,:) = [i LQ(i,:) K(:).' J M.gv M.OS_theta M.ts_theta M.phi_peak M.ts_phi M.IAE_theta M.IAE_phi]; %#ok<AGROW>
        if mod(i,25)==0, fprintf('%s %d/%d bestJ=%.5g\n', variants{iv,1}, i, nS, min(R(:,8))); end
      end
      T = array2table(R,'VariableNames',{'i','lq_theta','lq_thetad','lq_phid','K1','K2','K3','J','gv','OS_theta','ts_theta','phi_peak','ts_phi','IAE_theta','IAE_phi'});
      writetable(T,out);
    end
  case 'compare'
    pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
    recovered = @(o) ~o.diverged && isfinite(o.IAEtheta) && max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && max(abs(o.phi(o.t > o.t(end)-10))) < 0.05;
    rows = {};
    Kbest = struct();
    for iv = 1:size(variants,1)
      S = readtable(fullfile(Rdir,sprintf('E4_search_%s_%s.csv',variants{iv,1},tag)));
      [~,ib] = min(S.J); Kbest.(variants{iv,1}) = [S.K1(ib) S.K2(ib) S.K3(ib)];
      fprintf('best %s: K = [%g %g %g], J = %g, gv = %g (sample %d)\n', variants{iv,1}, Kbest.(variants{iv,1}), S.J(ib), S.gv(ib), S.i(ib));
    end
    inner = {'PSO-FLC','PSOFLC'; 'SMC-ESO','SMCESO'; 'LQR','LQR'};
    for rate = {'1kHz','100Hz'}
     for iv = 1:size(variants,1)
      for ii = 1:size(inner,1)
        for ip = 1:size(pert,1)+2
          P = P0; P.Kp_phi_mag = variants{iv,2}(1); P.Kv_phi_mag = variants{iv,2}(2); P.lqrK = Kbest.(variants{iv,1});
          if strcmp(rate{1},'100Hz'), P.ctrlTs = 1e-2; P.smcTs = 1e-2; else, P.ctrlTs = 1e-3; P.smcTs = 1e-3; end
          if ip <= size(pert,1)
            for f = 1:numel(pert{ip,2}), P.(pert{ip,2}{f}) = P0.(pert{ip,2}{f}) * pert{ip,3}{f}; end
            sc = 'S1'; cs = pert{ip,1}; kk = NaN;
          else, kk = [1.0 2.0]; kk = kk(ip-size(pert,1)); P.dist = struct('t',5.0,'dthetadot',kk); sc = 'S2'; cs = 'kick'; end
          o = sim_rwip_r5(inner{ii,2}, P, IC, 60, Ts); dv = isfield(o,'diverged') && o.diverged;
          rows(end+1,:) = {rate{1},variants{iv,1},inner{ii,1},sc,cs,kk,dv,gv(o,'IAEtheta'),gv(o,'IAEphi'),gv(o,'OS_theta'),gv(o,'ts_theta'),gv(o,'ts_phi'),gv(o,'Urms'),gv(o,'phi_end'),recovered(o)}; %#ok<AGROW>
          fprintf('%-5s %-9s %-8s %s %-9s k=%g div=%d IAEth=%.5g IAEphi=%.5g\n', rate{1}, variants{iv,1}, inner{ii,1}, sc, cs, kk, dv, gv(o,'IAEtheta'), gv(o,'IAEphi'));
        end
      end
     end
     T = cell2table(rows,'VariableNames',{'Rate','OuterGains','Inner','Scenario','Case','Kick','Diverged','IAEtheta','IAEphi','OS_theta','ts_theta','ts_phi','Urms','phi_end','Recovered'});
     writetable(T,fullfile(Rdir,['E4_compare_' tag '.csv']));
    end
  otherwise, error('stage must be search or compare');
end
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function dy = rhs3(y,V,Ip,Ir,bp,br,Ra,Kem,Kt,mgl)
th=y(1); thd=y(2); phd=y(3);
tau  = (Kt/Ra)*(V - Kem*phd);
thdd = (-tau + br*phd + mgl*sin(th) - bp*thd)/Ip;
phdd = (tau - br*phd)/Ir - thdd;
dy = [thd; thdd; phdd];
end
function v = gv(o,f), if isfield(o,f) && ~isempty(o.(f)), v = o.(f); else, v = NaN; end, end
function [J,M] = costJ(P, IC, Ts)
% PSO cost of pso_rwip_v13.m (constants copied, frozen 29 Oct 2026): constrained form, Tf = 30 s.
Tos=4.0; TendWin=8.0; thetaTol=0.002; phiTol=0.020; thetaHardLimit=1.2; thetaExceedPenalty=1e5; J_simError=1e14;
spec_OS=0.006; spec_tsTheta=3.0; spec_phiMax=0.52; spec_tsPhi=11.0; wCon=1e3; wCon2=1e4;
wEnd_theta=400; wEnd_phi=60; wSlope_phi=400; wPhiDotEnd=200; ru=0.002; rdu=2e-4; rtvdu=1e-4; rGain=1e-4;
M = struct('gv',NaN,'OS_theta',NaN,'ts_theta',NaN,'phi_peak',NaN,'ts_phi',NaN,'IAE_theta',NaN,'IAE_phi',NaN);
o = sim_rwip_r5('LQR', P, IC, 30, Ts);
if o.diverged || ~all(isfinite(o.theta)), J = J_simError; return, end
t=o.t; theta=o.theta; phi=o.phi; u=o.u; phid=o.phi_dot; dt = median(diff(t));
IAE_theta = trapz(t,abs(theta)); IAE_phi = trapz(t,abs(phi));
idxOS = t <= Tos;
if theta(1) >= 0, OS_theta = max(0, 0 - min(theta(idxOS))); else, OS_theta = max(0, max(theta(idxOS)) - 0); end
Eu = trapz(t,u.^2); du = [0; diff(u)]/dt; Edu = trapz(t,du.^2); tv_du = sum(abs(diff(du)));
idxEnd = t >= max(t(1), t(end)-TendWin);
theta_end = theta(end); phi_end = phi(end); phid_end = mean(abs(phid(idxEnd)));
tt = t(idxEnd); pp = phi(idxEnd); cf = [tt(:) ones(numel(tt),1)]\pp(:); phi_slope = cf(1);
ts_theta = lastAbove(t, theta, thetaTol); ts_phi = lastAbove(t, phi, phiTol); phi_peak = max(abs(phi));
gvv = max(0,OS_theta/spec_OS-1)+max(0,ts_theta/spec_tsTheta-1)+max(0,phi_peak/spec_phiMax-1)+max(0,ts_phi/spec_tsPhi-1);
Jobj = ru*Eu + rdu*Edu + rtvdu*tv_du + wEnd_theta*theta_end^2 + wEnd_phi*phi_end^2 + wSlope_phi*phi_slope^2 + wPhiDotEnd*phid_end^2 + rGain*0;
J = Jobj + wCon*gvv + wCon2*gvv^2;
if max(abs(theta)) > thetaHardLimit, J = J + thetaExceedPenalty; end
if ~isfinite(J), J = J_simError; end
M.gv=gvv; M.OS_theta=OS_theta; M.ts_theta=ts_theta; M.phi_peak=phi_peak; M.ts_phi=ts_phi; M.IAE_theta=IAE_theta; M.IAE_phi=IAE_phi;
end
function ts = lastAbove(t,e,tol), k = find(abs(e)>tol,1,'last'); if isempty(k), ts = 0; else, ts = t(k); end, end
