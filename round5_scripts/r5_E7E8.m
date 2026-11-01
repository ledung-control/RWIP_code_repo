function r5_E7E8()
% E7 (ESO bandwidth sweep) and E8 (oracle gain) -- protocol v1.0 section 4. Deterministic; uses the authors' SMCESO_cascade.m / sim_rwip_v13.m unchanged.
% E7: SMC-ESO, w0 in {10,15,20,30,40,60} rad/s, controller keeps NOMINAL g_theta_nom while the plant is perturbed.
% E8: same grid, g_theta_nom replaced by the TRUE perturbed gain (oracle).
% Check built in: w0 = 20, nominal gain must reproduce the stored Compare_v13_robust_results.mat SMC-ESO values (e.g. Kt -30 %: IAEtheta 5.4401).
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here))); cd(proj); addpath(proj);
Rdir = fullfile(here,'..','results'); Ldir = fullfile(here,'..','logs'); tag = ['MATLAB_' strrep(version('-release'),' ','')];
diary(fullfile(Ldir,sprintf('E7E8_%s_%s.txt',tag,datestr(now,'yyyymmdd_HHMMSS')))); diary on
fprintf('r5_E7E8 %s %s\n', datestr(now,31), version);
P0 = make_P_v13('fis_rwip_v13_best.fis'); IC=[0.10;0;0;0]; Ts=1e-3; Tf=60;
w0s = [10 15 20 30 40 60];
pert = { 'nominal',{},{}; 'mgl +30%',{'mgl_eq'},{1.30}; 'mgl -30%',{'mgl_eq'},{0.70}; 'Ip +30%',{'Ip_eq'},{1.30}; 'Ip -30%',{'Ip_eq'},{0.70}; 'Kt +30%',{'Kt'},{1.30}; 'Kt -30%',{'Kt'},{0.70}; 'adverse (mgl+30,Ip-30,Kt-30)',{'mgl_eq','Ip_eq','Kt'},{1.30,0.70,0.70} };
rows = {};
for variant = {'E7_nominal_gain','E8_oracle_gain'}
 for iw = 1:numel(w0s)
  for ip = 1:size(pert,1)
    P = P0; P.Kp_phi_mag = P.SMC_Kp_phi_mag; P.Kv_phi_mag = P.SMC_Kv_phi_mag;   % outer loop of the standalone SMC-ESO (applyOuter in Compare_v13_robust.m)
    P.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;                                   % controller's nominal gain (as Compare_v13_robust.m)
    P.G.w0 = w0s(iw);
    for f = 1:numel(pert{ip,2}), P.(pert{ip,2}{f}) = P0.(pert{ip,2}{f}) * pert{ip,3}{f}; end
    if strcmp(variant{1},'E8_oracle_gain'), P.g_theta_nom = -(P.Kt/P.Ra)/P.Ip_eq; end
    t0=tic; o = sim_rwip_v13('SMCESO', P, IC, Tf, Ts); dt=toc(t0);
    dv = isfield(o,'diverged') && o.diverged;
    chat = NaN; if isfield(o,'u') && numel(o.u)>2, chat = sqrt(mean(diff(o.u(:)).^2)); end
    rows(end+1,:) = {variant{1}, w0s(iw), pert{ip,1}, dv, gv(o,'IAEtheta'), gv(o,'IAEphi'), gv(o,'OS_theta'), gv(o,'ts_theta'), gv(o,'Urms'), chat, gv(o,'phi_end'), dt}; %#ok<AGROW>
    fprintf('%-16s w0=%2d %-30s div=%d IAEth=%.5g IAEphi=%.5g Urms=%.4g rmsDu=%.4g\n', variant{1}, w0s(iw), pert{ip,1}, dv, gv(o,'IAEtheta'), gv(o,'IAEphi'), gv(o,'Urms'), chat);
  end
 end
end
T = cell2table(rows,'VariableNames',{'Experiment','w0','Case','diverged','IAEtheta','IAEphi','OS_theta','ts_theta','Urms','rms_du_chatter','phi_end','wall_s'});
writetable(T, fullfile(Rdir, ['E7E8_' tag '.csv']));
fprintf('DONE %s\n', datestr(now,31)); diary off
end
function v = gv(o,f), if isfield(o,f) && ~isempty(o.(f)), v = double(o.(f)(1)); else, v = NaN; end, end
