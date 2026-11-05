%% Test_LongHorizon_v13.m -- On dinh dai han va tich luy troi
%
% ===========================================================================
% THAY THE CHO Test_LongHorizon.m (goi init_rwip_final, dung FIS cu)
% ---------------------------------------------------------------------------
% CAU HOI CAN TRA LOI
%   (1) TROI: sau 600 s, theta va phi co giu duoc 0 khong, hay tich luy dan?
%       Day la cho ma bo dieu khien cu that bai nang nhat: phi chay ramp toi
%       -76 rad trong 30 s, va cang chay lau cang xa.
%   (2) CHU TRINH GIOI HAN: bien do dao dong trong 100 s cuoi co bang 0 khong,
%       hay ton tai dao dong tu duy tri? (kien truc cu co chu trinh gioi han
%       +-1.1 rad chu ky 2.63 s -- chi lo ra khi chay du lau)
%   (3) TICH LUY SO: nang luong dieu khien trong 100 s cuoi phai ~0. Neu khong,
%       bo dieu khien van dang "lam viec" o trang thai xac lap -> hao mon that.
%   (4) NHIEU LAP LAI: chiu duoc bao nhieu cu hich lien tiep ma khong tich luy?
%
% Muc (4) quan trong voi RWIP hon cac he khac: banh da la mot khau TICH PHAN
% KEP tren bien can dieu tiet, nen sai so du sau moi lan nhieu co the cong don.
% ===========================================================================

clear; clc;
Ts = 1e-3;
Tf_drift  = 600;                 % 10 phut
Tf_rep    = 200;
kickTimes = 20:20:180;           % nhieu lap lai moi 20 s
kickAmp   = 1.0;                 % rad/s

P0 = make_P_v13('fis_rwip_v13_best.fis');
assert(isfile(P0.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P0.fisFile);
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;

modes = {'PSOFLC','SMCESO','PSOFLC_SMCESO'};
names = {'PSO-FLC v13','SMC-ESO cascade','PSO-FLC + SMC-ESO'};
mkP = @(m) outerFor(P0, m);

%% =====================================================================
%  PHAN 1 -- TROI DAI HAN (600 s, khong nhieu)
%% =====================================================================
fprintf('===== PHAN 1: TROI DAI HAN (%g s, khong nhieu) =====\n', Tf_drift);
fprintf('%-22s%12s%12s%14s%14s%12s\n', '', 'theta(T)', 'phi(T)', ...
        'max|th| 100s', 'max|phi| 100s', 'Eu 100s');
R1 = cell(1,3);
for m = 1:3
    o = sim_rwip_v13(modes{m}, mkP(modes{m}), [0.10;0;0;0], Tf_drift, Ts);
    R1{m} = o;
    if o.diverged, fprintf('%-22s%s\n', names{m}, '  MAT ON DINH'); continue; end
    late = o.t > Tf_drift - 100;
    fprintf('%-22s%12.3e%12.3e%14.3e%14.3e%12.3e\n', names{m}, ...
        o.theta(end), o.phi(end), max(abs(o.theta(late))), ...
        max(abs(o.phi(late))), trapz(o.t(late), o.u(late).^2));
end
fprintf(['\nDoc: bon cot cuoi deu phai ~0. Cot "max|.| 100s" khac 0 dang ke\n' ...
         '=> ton tai chu trinh gioi han. Cot "Eu 100s" khac 0 => bo dieu khien\n' ...
         '   van dang lam viec o xac lap (hao mon co cau, tieu thu dien).\n']);

%% =====================================================================
%  PHAN 2 -- BAT BIEN BAO TOAN o thoi gian dai
%% =====================================================================
fprintf('\n===== PHAN 2: BAT BIEN BAO TOAN (T = %g s) =====\n', Tf_drift);
inv_th = -P0.bp*0.10/P0.mgl_eq;
fprintf('Ly thuyet  (Ir*phidot(T) - bp*theta0)/mgl_eq voi phidot(T)=0 : %.5e\n', inv_th);
for m = 1:3
    if isempty(R1{m}) || R1{m}.diverged, continue; end
    o = R1{m};
    val = trapz(o.t, sin(o.theta));
    fprintf('  %-22s %12.5e   lech %.2f%%\n', names{m}, val, abs(val/inv_th-1)*100);
end

%% =====================================================================
%  PHAN 3 -- NHIEU LAP LAI (co tich luy khong?)
%% =====================================================================
fprintf('\n===== PHAN 3: %d CU HICH LIEN TIEP, moi %g s, bien do %.1f rad/s =====\n', ...
        numel(kickTimes), kickTimes(2)-kickTimes(1), kickAmp);
fprintf('%-22s%14s%14s%14s%14s\n','','dinh|th| lan 1','dinh|th| lan cuoi', ...
        'phi truoc cuoi','phi(T)');
R3 = cell(1,3);
for m = 1:3
    P = mkP(modes{m});
    P.distSeq = struct('t', kickTimes, 'dthetadot', kickAmp);   % xem ghi chu duoi
    o = simRepeated(modes{m}, P, Tf_rep, Ts, kickTimes, kickAmp);
    R3{m} = o;
    if isempty(o), fprintf('%-22s%s\n', names{m}, '  MAT ON DINH'); continue; end
    p1 = peakAfter(o, kickTimes(1),   kickTimes(2)-kickTimes(1));
    pN = peakAfter(o, kickTimes(end), 20);
    iPre = find(o.t >= kickTimes(end)-0.5, 1, 'first');
    fprintf('%-22s%14.4f%14.4f%14.4f%14.3e\n', names{m}, p1, pN, o.phi(iPre), o.phi(end));
end
fprintf(['\nDoc: "dinh lan cuoi" phai xap xi "dinh lan 1". Neu tang dan =>\n' ...
         '   co tich luy giua cac lan nhieu, bo dieu khien khong reset duoc\n' ...
         '   trang thai banh da giua cac su kien.\n']);

save('Test_LongHorizon_v13_results.mat','R1','R3','names','modes','P0', ...
     'Tf_drift','Tf_rep','kickTimes','kickAmp');
fprintf('\nDa luu Test_LongHorizon_v13_results.mat\n');

%% ---------------- Do thi ----------------
figure('Name','Dai han','Position',[80 80 1050 640]);
subplot(2,2,1); hold on; grid on
for m=1:3, if ~isempty(R1{m}) && ~R1{m}.diverged, plot(R1{m}.t, R1{m}.theta,'LineWidth',1.1); end, end
xlabel('t (s)'); ylabel('\theta (rad)'); title('Troi 600 s'); legend(names,'Location','best');
subplot(2,2,2); hold on; grid on
for m=1:3, if ~isempty(R1{m}) && ~R1{m}.diverged, plot(R1{m}.t, R1{m}.phi,'LineWidth',1.1); end, end
xlabel('t (s)'); ylabel('\phi (rad)'); title('Troi banh da');
subplot(2,2,3); hold on; grid on
for m=1:3, if ~isempty(R3{m}), plot(R3{m}.t, R3{m}.theta,'LineWidth',1.1); end, end
xlabel('t (s)'); ylabel('\theta (rad)'); title('Nhieu lap lai');
subplot(2,2,4); hold on; grid on
for m=1:3, if ~isempty(R3{m}), plot(R3{m}.t, R3{m}.phi,'LineWidth',1.1); end, end
xlabel('t (s)'); ylabel('\phi (rad)'); title('Banh da duoi nhieu lap lai');

%% =====================================================================
function P = outerFor(P, mode)
if strcmpi(mode,'SMCESO')
    P.Kp_phi_mag = P.SMC_Kp_phi_mag;  P.Kv_phi_mag = P.SMC_Kv_phi_mag;
end
end

function o = simRepeated(mode, P, Tf, Ts, tK, amp)
% sim_rwip_v13 chi ho tro MOT xung, nen ghep nhieu doan lien tiep: chay toi
% moc nhieu, cong xung vao theta_dot, roi chay tiep tu trang thai do.
o = [];  bnds = [0 tK Tf];
x  = [0.10;0;0;0];  T = []; TH=[]; THD=[]; PH=[]; PHD=[]; U=[];
for k = 1:numel(bnds)-1
    dur = bnds(k+1) - bnds(k);
    if dur <= 0, continue; end
    Pk = P; if isfield(Pk,'dist'), Pk = rmfield(Pk,'dist'); end
    s = sim_rwip_v13(mode, Pk, x, dur, Ts);
    if s.diverged, return; end
    T = [T; s.t + bnds(k)]; TH=[TH; s.theta]; THD=[THD; s.theta_dot];
    PH=[PH; s.phi]; PHD=[PHD; s.phi_dot]; U=[U; s.u];
    x = [s.theta(end); s.theta_dot(end); s.phi(end); s.phi_dot(end)];
    if k < numel(bnds)-1, x(2) = x(2) + amp; end   % cong xung
end
o = struct('t',T,'theta',TH,'theta_dot',THD,'phi',PH,'phi_dot',PHD,'u',U,'diverged',false);
end

function p = peakAfter(o, t0, win)
idx = o.t >= t0 & o.t <= t0 + win;
if ~any(idx), p = NaN; else, p = max(abs(o.theta(idx))); end
end
