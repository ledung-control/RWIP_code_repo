%% Benchmark_ComputeTime_v13.m -- Chi phi tinh toan cua vong trong
%
% ===========================================================================
% THAY THE CHO Benchmark_ComputeTime.m (do evalfis tren FIS cu + SMC goc)
% ---------------------------------------------------------------------------
% CANH BAO PHUONG PHAP -- doc truoc khi trich dan so lieu
%
% Nhieu bai bao so sanh chi phi FLC voi SMC bang cach do evalfis() cua Fuzzy
% Logic Toolbox. Do la phep do KHONG CONG BANG, vi hai ly do:
%
%  (1) evalfis() la ham DUNG CHUNG: no phan tich cau truc FIS, cap phat bo nho,
%      kiem tra dau vao, va suy dien Mamdani day du (25 luat x roi rac hoa dau
%      ra) o MOI lan goi. Khong ai trien khai FLC nhu vay tren vi dieu khien.
%  (2) Cach trien khai THUC TE la tinh truoc MAT DIEU KHIEN mot lan (offline)
%      roi tra bang + noi suy song tuyen luc chay. Voi FIS 2 vao 1 ra tinh
%      (khong doi theo thoi gian), hai cach cho ket qua GIONG NHAU trong sai
%      so noi suy, nhung chi phi chenh nhau vai bac.
%
% Script nay do CA HAI va bao cao rieng, de nguoi doc thay ro dau la chi phi
% cua THUAT TOAN va dau la chi phi cua CACH TRIEN KHAI.
%
% LUU Y THU HAI: thoi gian do bang MATLAB KHONG dai dien cho vi dieu khien.
% MATLAB co overhead thong dich lon hon nhieu so voi phan tinh toan thuc. Vi
% vay script cung bao cao UOC LUONG SO PHEP TINH (flop) -- do la dai luong
% chuyen doi duoc sang nen tang khac, con thoi gian MATLAB thi khong.
% ===========================================================================

clear; clc;
P0 = make_P_v13('fis_rwip_v13_best.fis');
assert(isfile(P0.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P0.fisFile);
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;

Ts = 1e-3;                 % chu ky lay mau khi trien khai
N  = 20000;                % so lan goi khi do
rng(0);

fis = readfis(P0.fisFile);
NG  = 201;
g   = linspace(-1,1,NG);
[A,B] = ndgrid(g,g);
tBuild = tic;
SURF = reshape(evalfis(fis,[A(:) B(:)]), NG, NG);
tBuildTable = toc(tBuild);

% Dau vao ngau nhien trong universe, dung CHUNG cho moi phep do
U1 = 2*rand(N,1)-1;  U2 = 2*rand(N,1)-1;

fprintf('=== Xay bang tra (mot lan, offline) ===\n');
fprintf('  luoi %dx%d, thoi gian dung bang: %.3f s\n', NG, NG, tBuildTable);
fprintf('  bo nho: %.1f KB (double) / %.1f KB (single)\n\n', ...
        NG*NG*8/1024, NG*NG*4/1024);

%% ---------------- 1. evalfis, goi tung mau (nhu trong vong dieu khien) ----
t = tic;
for k = 1:N, evalfis(fis, [U1(k) U2(k)]); end
t_evalfis = toc(t)/N;

%% ---------------- 2. Tra bang + noi suy song tuyen ----------------
t = tic;
for k = 1:N, lookup2(g, SURF, U1(k), U2(k)); end
t_lookup = toc(t)/N;

%% ---------------- 3. SMC-ESO mot buoc ----------------
z = [0;0;0]; gth = P0.g_theta_nom; G = P0.G;
t = tic;
for k = 1:N, [~, z] = SMCESO_cascade(U1(k)*0.1, 0, z, gth, Ts, G); end
t_smc = toc(t)/N;

%% ---------------- 4. Vong ngoai cascade (chung cho ca hai) ----------------
Kp = P0.Kp_phi_mag; Kv = P0.Kv_phi_mag; tm = P0.theta_ref_max;
t = tic;
for k = 1:N
    tr = -(Kp*U1(k) + Kv*U2(k));
    tr = max(-tm, min(tm, tr)); %#ok<NASGU>
end
t_outer = toc(t)/N;

%% ---------------- Bao cao thoi gian ----------------
fprintf('=== Thoi gian mot lan goi (MATLAB, trung binh %d lan) ===\n', N);
fprintf('  %-34s %10.3f us\n', 'evalfis() truc tiep',        t_evalfis*1e6);
fprintf('  %-34s %10.3f us\n', 'tra bang + noi suy',         t_lookup*1e6);
fprintf('  %-34s %10.3f us\n', 'SMC-ESO mot buoc',           t_smc*1e6);
fprintf('  %-34s %10.3f us\n', 'vong ngoai cascade',         t_outer*1e6);
fprintf('  ---\n');
fprintf('  %-34s %10.1fx\n', 'evalfis / tra bang',           t_evalfis/t_lookup);
fprintf('  %-34s %10.2fx\n', 'tra bang / SMC-ESO',           t_lookup/t_smc);

fprintf('\n=== Chi phi vong dieu khien day du, Ts = %g s ===\n', Ts);
cfgs = { 'PSO-FLC v13 (tra bang)',   t_lookup + t_outer ;
         'PSO-FLC v13 (evalfis)',    t_evalfis + t_outer ;
         'SMC-ESO cascade',          t_smc + t_outer ;
         'PSO-FLC + SMC-ESO',        t_lookup + t_smc + t_outer };
fprintf('  %-30s%12s%14s\n','','us/buoc','%% CPU @1kHz');
for i = 1:size(cfgs,1)
    fprintf('  %-30s%12.3f%13.2f%%\n', cfgs{i,1}, cfgs{i,2}*1e6, 100*cfgs{i,2}/Ts);
end

%% ---------------- Uoc luong so phep tinh (chuyen doi duoc) ----------------
fprintf('\n=== Uoc luong so phep tinh moi buoc (dai luong chuyen doi duoc) ===\n');
flop_lookup = 20;    % 2 clamp, 2 scale, 2 floor, 2 frac, 4 doc bang, 8 nhan, 3 cong
flop_evalfis = 25*4 + 5*101 + 2*101;   % 25 luat x (2 MF + min + imp) + hop thanh + trong tam
flop_smc  = 25;      % s, sat, luat dieu khien (~6), ESO 3 trang thai (~12)
flop_outer = 5;
fprintf('  %-34s ~%5d flop\n', 'tra bang + noi suy',  flop_lookup);
fprintf('  %-34s ~%5d flop\n', 'evalfis (Mamdani day du)', flop_evalfis);
fprintf('  %-34s ~%5d flop\n', 'SMC-ESO mot buoc',    flop_smc);
fprintf('  %-34s ~%5d flop\n', 'vong ngoai',          flop_outer);
fprintf('  ---\n');
fprintf('  %-34s ~%5d flop\n', 'PSO-FLC (tra bang) tong',  flop_lookup+flop_outer);
fprintf('  %-34s ~%5d flop\n', 'SMC-ESO cascade tong',     flop_smc+flop_outer);
fprintf('  %-34s ~%5d flop\n', 'PSO-FLC + SMC-ESO tong',   flop_lookup+flop_smc+flop_outer);
fprintf(['\n  => Khi trien khai dung cach, PSO-FLC va SMC-ESO co chi phi CUNG BAC.\n' ...
         '     Chenh lech hang tram lan ma cac bai bao thuong bao cao la chi phi\n' ...
         '     cua evalfis(), tuc cua CACH TRIEN KHAI, khong phai cua thuat toan.\n']);

%% ---------------- Do phan giai bang tra can bao nhieu? ----------------
fprintf('\n=== Anh huong cua do phan giai bang tra len chat luong dieu khien ===\n');
fprintf('  %-12s%10s%12s%12s%12s%12s\n','luoi','KB','IAEtheta','IAEphi','vot lo','ts_theta');
gridList = [26 51 101 201 401];
ref = [];
for ng = gridList
    Pg = P0;  Pg.surfN = ng;                 % sim_rwip_v13 doc truong nay neu co
    o = sim_rwip_v13('PSOFLC', Pg, [0.10;0;0;0], 60, 1e-3);
    if isempty(ref), refIdx = find(gridList==401,1); end %#ok<NASGU>
    fprintf('  %-12s%10.1f%12.5f%12.4f%12.6f%12.3f\n', sprintf('%dx%d',ng,ng), ...
            ng*ng*8/1024, o.IAEtheta, o.IAEphi, o.OS_theta, o.ts_theta);
end
fprintf(['\n  => Chon do phan giai nho nhat ma chi so con on dinh. Bo nho bang tra\n' ...
         '     ty le voi BINH PHUONG do phan giai, nen giam tu 201 xuong 101 tiet\n' ...
         '     kiem 4 lan bo nho.\n']);

save('Benchmark_ComputeTime_v13_results.mat', ...
     't_evalfis','t_lookup','t_smc','t_outer','N','Ts','NG','tBuildTable','gridList');
fprintf('\nDa luu Benchmark_ComputeTime_v13_results.mat\n');

%% =====================================================================
function v = lookup2(g, S, a, b)
n = numel(g); h = 2/(n-1);
a = max(-1,min(1,a)); b = max(-1,min(1,b));
xa = (a+1)/h; xb = (b+1)/h;
i = min(max(floor(xa),0), n-2); j = min(max(floor(xb),0), n-2);
fa = xa-i; fb = xb-j; i=i+1; j=j+1;
v = S(i,j)*(1-fa)*(1-fb) + S(i+1,j)*fa*(1-fb) ...
  + S(i,j+1)*(1-fa)*fb   + S(i+1,j+1)*fa*fb;
end
