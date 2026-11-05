%% Compare_v13.m -- So sanh SONG PHANG PSO-FLC v13 vs SMC-ESO cascade
%
% ===========================================================================
% TAI SAO PHEP SO SANH CU KHONG SONG PHANG
% ---------------------------------------------------------------------------
% SMC-ESO ban goc dung mat truot  s = c*theta + omega_p  -- KHONG chua phi.
% No on dinh con lac bang cach do mo-men dong luong vao banh da roi de do:
%   Eval_PSOFLC_vs_SMCESO_results.mat, 60 s:  RMSEphi = 10.02 rad, IAEphi = 523
% Tuc no giai mot bai toan DE HON (2 trang thai thay vi 4). So no voi mot bo
% co dieu khien banh da tren chi so theta la khong cong bang.
%
% Script nay cho CA HAI bo dung CHUNG vong ngoai cascade, CHUNG doi tuong,
% CHUNG dieu kien dau -- chi khac luat vong trong (mo so vs truot).
%
% ---------------------------------------------------------------------------
% BAT BIEN BAO TOAN (co so ly thuyet cua toan bo so sanh)
% ---------------------------------------------------------------------------
% Cong tich phan hai phuong trinh dong luc tren [0,T], voi theta_dot(0)=
% theta_dot(T)=0 va phi_dot(0)=0:
%
%       Int_0^T sin(theta) dt  =  ( Ir*phi_dot(T) - bp*theta_0 ) / mgl_eq
%
% He qua: BAT KY bo dieu khien nao dua banh da ve nghi (phi_dot(T)=0) deu buoc
% phai cho theta di qua diem dat -- vot lo AM la KHONG THE TRANH. Do khong phai
% khiem khuyet chinh dinh ma la tinh chat cau truc cua con lac banh da phan luc.
% Script in ra cot IntSinTheta de kiem chung: gia tri phai GIONG NHAU cho moi
% bo dieu khien (sai lech chi do phi_dot(T) du con lai).
% ===========================================================================

clear; clc;
Tf = 60;  Ts = 1e-3;
IC = [0.10; 0; 0; 0];

% FIS phai KHOP voi bo gain trong make_P_v13. Ca hai deu den tu cung mot lan
% chay pso_rwip_v13 -- ghep gain moi voi FIS cu la loi da gap nhieu lan.
P = make_P_v13('fis_rwip_v13_best.fis');
assert(isfile(P.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P.fisFile);

% Doi chieu voi ket qua PSO de bat ngay moi lech ghep noi.
if isfile('pso_rwip_v13_result.mat')
    R0 = load('pso_rwip_v13_result.mat');
    g = R0.GBestPos;
    chk = [P.Ke P.Kde P.Ku P.Kd_theta P.Kd_phi P.Kp_phi_mag P.Kv_phi_mag];
    ref = [g(4) g(5) g(6) g(7) g(8) g(9) g(10)];
    d   = max(abs(chk-ref)./max(abs(ref),eps));
    if d > 1e-6
        warning(['Gain trong make_P_v13 LECH %.2g%% so voi pso_rwip_v13_result.mat.\n' ...
                 'Cap nhat make_P_v13.m truoc khi bao cao so lieu.'], 100*d);
    else
        fprintf('Gain khop pso_rwip_v13_result.mat  [OK]\n');
    end
end

% PSO-FLC: dung vong ngoai da chinh dinh cho no
Pf = P;
% SMC-ESO: dung vong ngoai da chinh dinh RIENG cho no
Ps = P;  Ps.Kp_phi_mag = P.SMC_Kp_phi_mag;  Ps.Kv_phi_mag = P.SMC_Kv_phi_mag;

% PSO-FLC + RL (BC-only): tuy chon, chi chay neu file agent ton tai. Doi
% RLBC_FILE de chuyen sang seed khac hoac agent_bc_seedN.mat (sau TD3) khi co.
RLBC_FILE = 'rl_v13_bc/agent_bconly_seed1.mat';
runRL = isfile(RLBC_FILE);
if runRL
    Pr = rl_policy_from_agent(P, RLBC_FILE);
else
    fprintf(2, 'Khong thay %s -- bo qua cau hinh PSO-FLC + RL.\n', RLBC_FILE);
end

fprintf('Dang chay %d cau hinh (Tf=%g s, Ts=%g s)...\n', 3+runRL, Tf, Ts);
R = struct();
R.PSOFLC = sim_rwip_v13('PSOFLC',        Pf, IC, Tf, Ts);
R.SMCESO = sim_rwip_v13('SMCESO',        Ps, IC, Tf, Ts);
R.HYBRID = sim_rwip_v13('PSOFLC_SMCESO', Pf, IC, Tf, Ts);

names = {'PSO-FLC v13','SMC-ESO cascade','PSO-FLC + SMC-ESO'};
keys  = {'PSOFLC','SMCESO','HYBRID'};
if runRL
    R.RLBC = sim_rwip_v13('PSOFLC_RL', Pr, IC, Tf, Ts);
    names{end+1} = 'PSO-FLC + RL (BC-only)';
    keys{end+1}  = 'RLBC';
end

%% ---------------- Bang chi so ----------------
hdr = {'IAEtheta','IAEphi','IAEu','RMSEtheta','RMSEphi','Urms', ...
       'OS_theta','ts_theta','ts_phi','phi_end','phidot_end','Edu'};
fprintf('\n%-20s', 'Chi so');
for k=1:numel(names), fprintf('%18s', names{k}); end
fprintf('\n%s\n', repmat('-',1,20+18*numel(names)));
for h = 1:numel(hdr)
    fprintf('%-20s', hdr{h});
    for k = 1:numel(keys), fprintf('%18.5g', R.(keys{k}).(hdr{h})); end
    fprintf('\n');
end

fprintf('\n%-20s', 'IntSinTheta');
for k=1:numel(keys), fprintf('%18.4e', R.(keys{k}).IntSinTheta); end
fprintf('\n  (bat bien bao toan -- phai gan bang nhau o moi bo dieu khien)\n');

%% ---------------- Ai thang chi so nao ----------------
fprintf('\n=== PSO-FLC v13 so voi SMC-ESO cascade ===\n');
% NGUONG "COI NHU BANG 0": duoi muc nay, ty le giua hai gia tri khong con y
% nghia vat ly (vd phi_end 5e-8 vs 2e-19 -> "tot hon 236 trieu lan" la vo nghia
% -- CA HAI deu bang 0 trong moi thang do thuc te). Nguong dat theo do phan giai
% do luong hop ly cua encoder/IMU.
NEG = struct('phi_end',1e-4, 'phidot_end',1e-4, 'OS_theta',1e-5, ...
             'IAEtheta',1e-5, 'IAEphi',1e-4, 'RMSEtheta',1e-5, 'RMSEphi',1e-5);
for h = 1:numel(hdr)
    key = hdr{h};
    a = R.PSOFLC.(key); b = R.SMCESO.(key);
    if any(strcmp(key,{'phi_end','phidot_end'})), a=abs(a); b=abs(b); end
    thr = 0; if isfield(NEG,key), thr = NEG.(key); end
    if a < thr && b < thr
        fprintf('  %-12s  %12.5g  vs %12.5g   -> CA HAI ~ 0 (duoi nguong %g), khong so sanh\n', ...
                key, a, b, thr);
        continue
    end
    if a < b, w='PSO-FLC'; r=b/max(a,eps); else, w='SMC-ESO'; r=a/max(b,eps); end
    fprintf('  %-12s  %12.5g  vs %12.5g   -> %-8s tot hon %6.1f lan\n', key, a, b, w, r);
end

%% ---------------- Cong muc 6: PSO-FLC + RL vs PSO-FLC thuan (danh nghia) ---
% Dieu kien phu (muc 6 THIETKE_MOITRUONG_RL_v13.md): "khong xau hon PSO-FLC
% thuan qua 20% o BAT KY chi so nao trong Bang 1 (danh nghia)". Kiem TOAN BO
% cot hdr, khong chi IAEtheta -- dung tinh than "khong xau hon" tren dien rong.
%
% TU KIEM (them sau khi ra soat Compare_v13_robust.m / Test_Disturbance_v13.m):
% hai file do dung CHI SO VI TRI vao mang modes/names co thu tu thay doi duoc
% (vd degr(2), kickMax(2)) tro nham vao 'SMCESO' doc lap thay vi 'PSOFLC_SMCESO'
% -- moc sai, lam mot cong "DAT" gia. Cong nay o day dung TEN TRUONG CO DINH
% (R.PSOFLC, R.RLBC), KHONG dung chi so vi tri, nen ve nguyen tac mien nhiem
% voi cung loai loi. Assert duoi kiem dieu do dung -- neu R.PSOFLC vo tinh bi
% gan lai bang R.SMCESO/R.HYBRID o dau do phia tren, cong se BAO LOI thay vi
% am tham cham sai moc.
assert(~isequal(R.PSOFLC.IAEtheta, R.SMCESO.IAEtheta), ...
    'Tu kiem that bai: R.PSOFLC trung R.SMCESO -- co the bi gan nham.');
assert(~isequal(R.PSOFLC.IAEtheta, R.HYBRID.IAEtheta), ...
    'Tu kiem that bai: R.PSOFLC trung R.HYBRID -- co the bi gan nham.');
if runRL
    fprintf('\n=== Cong muc 6: PSO-FLC + RL (BC-only) so voi PSO-FLC thuan (danh nghia) ===\n');
    fprintf('  (dang so R.PSOFLC voi R.RLBC bang ten truong co dinh -- da tu kiem KHONG trung SMCESO/HYBRID)\n');
    worst = 0; worstKey = '';
    for h = 1:numel(hdr)
        key = hdr{h}; a = R.PSOFLC.(key); b = R.RLBC.(key);
        thr = 0; if isfield(NEG,key), thr = NEG.(key); end
        if any(strcmp(key,{'phi_end','phidot_end'})), a=abs(a); b=abs(b); end
        if a < thr && b < thr
            fprintf('  %-12s  ca hai ~ 0, bo qua\n', key); continue
        end
        d = 100*(b-a)/max(abs(a),eps);
        fprintf('  %-12s  PSO-FLC %12.5g  RL %12.5g   lech %+7.1f%%\n', key, a, b, d);
        if abs(d) > abs(worst), worst = d; worstKey = key; end
    end
    fprintf('  rho (ty le can thiep, tham chieu SMC-ESO 0.28-0.43) = %.3f\n', R.RLBC.rho);
    if abs(worst) <= 20
        fprintf('  => DAT dieu kien phu muc 6 (lech lon nhat %+.1f%% o %s, <= 20%%)\n', worst, worstKey);
    else
        fprintf(2, '  => KHONG DAT dieu kien phu muc 6 (lech lon nhat %+.1f%% o %s, > 20%%)\n', worst, worstKey);
    end
end

% --- Edu tach rieng qua do khoi dau ---
% Edu tren toan hanh trinh bi CHI PHOI boi buoc nhay dien ap tai t=0 (theta0=0.10
% doi ngay dien ap khac 0). Do la he qua cua DIEU KIEN DAU, khong phai dac tinh
% chattering cua bo dieu khien. Bao cao ca hai de tranh hieu nham.
fprintf('\n=== Edu: toan bo vs bo qua 0.1 s dau (tach do khoi dau khoi chattering) ===\n');
for k = 1:numel(keys)
    r = R.(keys{k}); dt = r.t(2)-r.t(1);
    du = [0; diff(r.u)]/dt; m = r.t > 0.1;
    fprintf('  %-20s Edu_toanbo = %10.1f   Edu_sau_0.1s = %10.1f\n', ...
            names{k}, r.Edu, trapz(r.t(m), du(m).^2));
end

%% ---------------- Do thi ----------------
figure('Name','Compare v13','Position',[80 80 1150 760]);
sp = {'theta','phi','u','phi_dot'};
ylb = {'\theta (rad)','\phi (rad)','V (V)','$\dot\phi$ (rad/s)'};
for i = 1:4
    subplot(2,2,i); hold on; grid on
    for k = 1:numel(keys)
        plot(R.(keys{k}).t, R.(keys{k}).(sp{i}), 'LineWidth', 1.2);
    end
    if i==1, yline(0,'k:'); end
    xlabel('t (s)');
    if i==4, ylabel('$\dot\phi$ (rad/s)','Interpreter','latex');
    else,    ylabel(ylb{i}); end
    if i==1, legend(names,'Location','best'); xlim([0 20]); end
    if i==2, xlim([0 30]); end
    if i==3, xlim([0 10]); end
end
sgtitle('PSO-FLC v13 vs SMC-ESO cascade -- cung vong ngoai, cung doi tuong');

% Hinh rieng: phong to vung vot lo (bang chung cua dinh luat bao toan)
figure('Name','Vot lo -- dinh luat bao toan','Position',[120 120 700 420]);
hold on; grid on
for k = 1:numel(keys)
    plot(R.(keys{k}).t, R.(keys{k}).theta, 'LineWidth', 1.4);
end
yline(0,'k:'); xlim([0 12]); ylim([-0.06 0.11]);
xlabel('t (s)'); ylabel('\theta (rad)');
title('CA HAI bo deu vot lo am -- he qua cua bat bien bao toan');
legend(names,'Location','best');

if runRL
    save('Compare_v13_results.mat','R','P','Pf','Ps','Pr','IC','Tf','Ts','names','keys','RLBC_FILE','runRL');
else
    save('Compare_v13_results.mat','R','P','Pf','Ps','IC','Tf','Ts','names','keys','runRL');
end
fprintf('\nDa luu Compare_v13_results.mat\n');
