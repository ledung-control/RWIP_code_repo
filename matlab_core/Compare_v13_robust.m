%% Compare_v13_robust.m -- Thu nghiem BEN VUNG: sai lech tham so + nhieu ngoai
%
% ===========================================================================
% TAI SAO CAN THU NGHIEM NAY
% ---------------------------------------------------------------------------
% O kich ban DANH NGHIA, lop hieu chinh SMC-ESO dong gop rat it (xem
% Compare_v13): bo lai kem PSO-FLC thuan o IAEphi, ts_phi va Edu, chi hon o
% vot lo. Nhung danh nghia KHONG phai noi truot + quan sat nhieu mo rong sinh
% ra de thang -- ly do ton tai cua chung la BEN VUNG truoc sai lech mo hinh
% va nhieu ngoai. Danh gia o dieu kien danh nghia roi ket luan la khong cong
% bang, dung kieu sai lam da chi ra o phep so sanh truoc.
%
% ---------------------------------------------------------------------------
% DIEM MAU CHOT VE TINH CONG BANG
% ---------------------------------------------------------------------------
% Khi nhieu loan tham so, DOI TUONG doi nhung BO DIEU KHIEN van giu gia tri
% DANH NGHIA -- no khong biet doi tuong da doi. Vi vay g_theta cua SMC duoc
% ghim bang P.g_theta_nom tinh tu tham so danh nghia. Neu de bo dieu khien
% dung tham so da nhieu loan thi hoa ra no duoc "mach nuoc", va phep thu ben
% vung mat het y nghia.
%
% KICH BAN 1: sai lech +-30% tren mgl_eq, Ip_eq, Kt (rieng le va ket hop xau nhat)
% KICH BAN 2: xung van toc tren theta_dot tai t=5 s, DO NHI PHAN tim bien do
%             lon nhat ma moi bo con phuc hoi duoc.
% ===========================================================================

clear; clc;
Tf = 60;  Ts = 1e-3;  IC = [0.10; 0; 0; 0];

P0 = make_P_v13('fis_rwip_v13_best.fis');
assert(isfile(P0.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P0.fisFile);

% Ghim he so control-affine DANH NGHIA cho bo dieu khien
P0.g_theta_nom = -(P0.Kt/P0.Ra)/P0.Ip_eq;

modes = {'PSOFLC','SMCESO','PSOFLC_SMCESO'};
names = {'PSO-FLC v13','SMC-ESO cascade','PSO-FLC + SMC-ESO'};

% PSO-FLC + RL (BC-only): tuy chon, chi them neu file agent ton tai.
RLBC_FILE = 'rl_v13_bc/agent_bconly_seed1.mat';
if isfile(RLBC_FILE)
    modes{end+1} = 'PSOFLC_RL';
    names{end+1} = 'PSO-FLC + RL (BC-only)';
else
    fprintf(2, 'Khong thay %s -- bo qua cau hinh PSO-FLC + RL.\n', RLBC_FILE);
end
nMode = numel(modes);

% Vong ngoai rieng cho SMC (nhu trong Compare_v13); RL dung CHUNG vong ngoai
% voi PSOFLC thuan nen khong can swap gain, chi can gan chinh sach.
mkP = @(base, mode) attachMode(applyOuter(base, mode), mode, RLBC_FILE);

%% =====================================================================
%  KICH BAN 1 -- SAI LECH THAM SO
%% =====================================================================
pert = { 'danh nghia',        {},                  {} ;
         'mgl_eq  +30%',      {'mgl_eq'},          {1.30} ;
         'mgl_eq  -30%',      {'mgl_eq'},          {0.70} ;
         'Ip_eq   +30%',      {'Ip_eq'},           {1.30} ;
         'Ip_eq   -30%',      {'Ip_eq'},           {0.70} ;
         'Kt      +30%',      {'Kt'},              {1.30} ;
         'Kt      -30%',      {'Kt'},              {0.70} ;
         'ket hop xau nhat',  {'mgl_eq','Ip_eq','Kt'}, {1.30,0.70,0.70} };

fprintf('===== KICH BAN 1: SAI LECH THAM SO (+-30%%) =====\n');
fprintf('%-20s', 'truong hop');
for k=1:nMode, fprintf('%26s', names{k}); end
fprintf('\n%s\n', repmat('-',1,20+26*nMode));

R1 = cell(size(pert,1),nMode);
for r = 1:size(pert,1)
    fprintf('%-20s', pert{r,1});
    for m = 1:nMode
        P = mkP(P0, modes{m});
        for f = 1:numel(pert{r,2})
            P.(pert{r,2}{f}) = P0.(pert{r,2}{f}) * pert{r,3}{f};
        end
        o = sim_rwip_v13(modes{m}, P, IC, Tf, Ts);
        R1{r,m} = o;
        if o.diverged || ~isfinite(o.IAEtheta)
            fprintf('%26s', 'MAT ON DINH');
        else
            fprintf('%12.4f%14.3f', o.IAEtheta, o.IAEphi);
        end
    end
    fprintf('\n');
end
fprintf('(moi o: IAEtheta  IAEphi)\n');

% --- Cong muc 6, phan "xuong cap lon nhat duoi sai lech +-30%" ----------
% Xuong cap = IAEtheta(truong hop nhieu loan) / IAEtheta(danh nghia CUNG mode)
% - 1, lay max tren cac truong hop 2..end. Moc: PSO-FLC +13%, SMC-ESO+9%
% (Bang 2, KETQUA_BAIBAO_v13.md). Dat neu PSO-FLC+RL <= moc SMC-ESO cascade.
iRL = find(strcmp(modes,'PSOFLC_RL'), 1);
% Muc 6 dinh nghia HAI moc rieng: "moc PSO-FLC thuan" va "moc bo lai SMC-ESO"
% (tuc PSOFLC_SMCESO, KHONG PHAI SMCESO doc lap). Tra theo TEN, khong dung so
% cung -- SMCESO doc lap la baseline yeu hon, biet truoc (vd +5185% o day),
% dung no lam moc se luon cho "DAT" gia. Day chinh la loai loi so-nham-baseline
% ma toan bo du an nay da nhieu lan canh bao (SMC-ESO goc khong dieu khien phi).
iHybrid = find(strcmp(modes,'PSOFLC_SMCESO'), 1);
if ~isempty(iRL) && ~isempty(iHybrid)
    fprintf('\nCong muc 6 (xuong cap lon nhat duoi sai lech +-30%%):\n');
    degr = nan(1,nMode);
    for m = 1:nMode
        base = R1{1,m};
        if base.diverged || ~isfinite(base.IAEtheta), continue; end
        worstDeg = 0;
        for r = 2:size(pert,1)
            o = R1{r,m};
            if o.diverged || ~isfinite(o.IAEtheta), worstDeg = inf; continue; end
            d = 100*(o.IAEtheta/base.IAEtheta - 1);
            worstDeg = max(worstDeg, d);
        end
        degr(m) = worstDeg;
        fprintf('  %-22s +%.1f%%\n', names{m}, worstDeg);
    end
    if isfinite(degr(iRL)) && degr(iRL) <= degr(iHybrid)
        fprintf('  => DAT moc nay (<= %.1f%% cua %s).\n', degr(iHybrid), names{iHybrid});
    else
        fprintf(2, '  => KHONG DAT moc nay (moc %s = %.1f%%).\n', names{iHybrid}, degr(iHybrid));
    end
end

%% =====================================================================
%  KICH BAN 2 -- NHIEU NGOAI, DO NHI PHAN NGUONG PHUC HOI
%% =====================================================================
fprintf('\n===== KICH BAN 2: XUNG VAN TOC TAI t=5 s =====\n');
fprintf('Tim bien do LON NHAT ma bo dieu khien con phuc hoi (do nhi phan).\n');
fprintf('Tieu chi phuc hoi: |theta| < 0.01 rad va |phi| < 0.05 rad trong 10 s cuoi.\n\n');

recovered = @(o) ~o.diverged && isfinite(o.IAEtheta) && ...
                 max(abs(o.theta(o.t > o.t(end)-10))) < 0.01 && ...
                 max(abs(o.phi  (o.t > o.t(end)-10))) < 0.05;

kickMax = zeros(1,nMode);
for m = 1:nMode
    P = mkP(P0, modes{m});
    loK = 0;  hiK = 8.0;                       % rad/s
    % kiem tra can tren truoc
    P.dist = struct('t',5.0,'dthetadot',hiK);
    if recovered(sim_rwip_v13(modes{m}, P, IC, Tf, Ts))
        kickMax(m) = hiK;
        fprintf('  %-20s > %.2f rad/s (vuot can tren thu nghiem)\n', names{m}, hiK);
        continue
    end
    for it = 1:12                              % 12 lan chia doi -> sai so ~0.002
        mid = 0.5*(loK+hiK);
        P.dist = struct('t',5.0,'dthetadot',mid);
        if recovered(sim_rwip_v13(modes{m}, P, IC, Tf, Ts)), loK = mid; else, hiK = mid; end
    end
    kickMax(m) = loK;
    fprintf('  %-20s nguong phuc hoi = %.3f rad/s\n', names{m}, loK);
end

fprintf('\n  Ty le so voi PSO-FLC thuan:');
for m=1:nMode, fprintf('  %s = %.2fx', names{m}, kickMax(m)/max(kickMax(1),eps)); end
fprintf('\n');

% --- Cong muc 6, phan "nguong nhieu phuc hoi" ---------------------------
% Moc dung la "bo lai SMC-ESO" (PSOFLC_SMCESO, 5.047 rad/s trong ban chinh
% thuc) -- KHONG PHAI SMCESO doc lap. Dung chinh kickMax(iHybrid) cua LAN CHAY
% NAY de khong lech so voi dieu kien mo phong hien tai.
iRL     = find(strcmp(modes,'PSOFLC_RL'), 1);
iHybrid = find(strcmp(modes,'PSOFLC_SMCESO'), 1);
if ~isempty(iRL) && ~isempty(iHybrid)
    fprintf('\n  Cong muc 6 (nguong nhieu phuc hoi): PSO-FLC+RL = %.3f | moc %s = %.3f rad/s\n', ...
            kickMax(iRL), names{iHybrid}, kickMax(iHybrid));
    if kickMax(iRL) >= kickMax(iHybrid)
        fprintf('  => DAT moc nay.\n');
    else
        fprintf(2, '  => KHONG DAT moc nay (thieu %.1f%%).\n', 100*(1-kickMax(iRL)/kickMax(iHybrid)));
    end
end

%% =====================================================================
%  Dap ung tai mot muc nhieu CHUNG (de ve do thi so sanh)
%% =====================================================================
kCommon = 0.9*min(kickMax);
fprintf('\n===== Dap ung tai xung CHUNG = %.3f rad/s =====\n', kCommon);
fprintf('%-22s%12s%12s%12s%12s\n','','IAEtheta','IAEphi','max|theta|','t_phuc_hoi');
R2 = cell(1,nMode);
for m = 1:nMode
    P = mkP(P0, modes{m});  P.dist = struct('t',5.0,'dthetadot',kCommon);
    o = sim_rwip_v13(modes{m}, P, IC, Tf, Ts);  R2{m} = o;
    idx = o.t >= 5.0;
    kk = find(abs(o.theta(idx))>0.002, 1, 'last');
    tRec = ternR(isempty(kk), 0, o.t(find(idx,1)+kk-1) - 5.0);
    fprintf('%-22s%12.4f%12.3f%12.4f%10.2f s\n', names{m}, ...
            o.IAEtheta, o.IAEphi, max(abs(o.theta(idx))), tRec);
end

%% ---------------- Do thi ----------------
figure('Name','Ben vung: nhieu ngoai','Position',[80 80 1000 620]);
subplot(2,1,1); hold on; grid on
for m=1:nMode, plot(R2{m}.t, R2{m}.theta, 'LineWidth',1.3); end
xlim([4 25]); xlabel('t (s)'); ylabel('\theta (rad)');
title(sprintf('Phuc hoi sau xung %.2f rad/s tai t = 5 s', kCommon));
legend(names,'Location','best');
subplot(2,1,2); hold on; grid on
for m=1:nMode, plot(R2{m}.t, R2{m}.phi, 'LineWidth',1.3); end
xlim([4 40]); xlabel('t (s)'); ylabel('\phi (rad)');

figure('Name','Ben vung: nguong nhieu','Position',[120 120 620 420]);
bar(kickMax); grid on
set(gca,'XTickLabel',names); ylabel('Xung lon nhat con phuc hoi (rad/s)');
title('Bien do nhieu ngoai toi da');

save('Compare_v13_robust_results.mat','R1','R2','pert','kickMax','names','modes','P0','kCommon','nMode');
fprintf('\nDa luu Compare_v13_robust_results.mat\n');

%% =====================================================================
function P = applyOuter(P, mode)
% SMC-ESO doc lap dung vong ngoai da chinh dinh RIENG cho no.
if strcmpi(mode,'SMCESO')
    P.Kp_phi_mag = P.SMC_Kp_phi_mag;
    P.Kv_phi_mag = P.SMC_Kv_phi_mag;
end
end

function P = attachMode(P, mode, rlbcFile)
% Gan chinh sach RL neu mode la PSOFLC_RL. Goi SAU applyOuter -- PSOFLC_RL
% dung vong ngoai CHUNG voi PSOFLC thuan nen applyOuter la no-op cho no.
if strcmpi(mode,'PSOFLC_RL')
    P = rl_policy_from_agent(P, rlbcFile);
end
end

function y = ternR(c,a,b)
if c, y=a; else, y=b; end
end
