%% Test_Disturbance_v13.m -- Phuc hoi sau nhieu ngoai, quet NHIEU DIEU KIEN DAU
%
% ===========================================================================
% THAY THE CHO Test_Disturbance_Rejection.m
% ---------------------------------------------------------------------------
% Script cu KHONG con dung duoc, vi ba ly do doc lap:
%   (1) Goi init_rwip_final -- tep da bi thay bang init_rwip_v13.
%   (2) Dung fis_rwip_best.fis va kien truc FLC cu (dau vao thu 2 la phi_dot,
%       gain nhan doi, FIS lech tam) -- xem 6 loi trong KE_HOACH_HOAN_THIEN_v13.
%   (3) Truc so sanh cua no la "baseline cu doi RL", ma agent TD3 trong
%       RWIP_TD3Agent_paper.mat duoc huan luyen tren moi truong chua bo dieu
%       khien nen hong -> phai huan luyen lai truoc khi so sanh co nghia.
%
% Ban nay giu lai y tuong co gia tri cua script cu -- quet nhieu DIEU KIEN DAU
% thay vi chi mot -- va bo phan RL de lai sau khi huan luyen lai.
%
% KHAC voi Compare_v13_robust.m: o do ta do NGUONG nhieu lon nhat con phuc hoi
% (mot con so cho moi bo dieu khien, tu mot dieu kien dau). O day ta co dinh
% bien do nhieu va quet dieu kien dau, de kiem tra ket qua co ON DINH tren
% toan dai lam viec hay chi dung o mot diem.
% ===========================================================================

clear; clc;
Tf = 40;  Ts = 1e-3;
t_dist = 5.0;                       % thoi diem ap nhieu (s)
kicks  = [0.3, 1.0, 2.0];           % bien do xung van toc (rad/s)
theta0_list = [-0.15, -0.08, 0.05, 0.10, 0.15];
recThresh   = 0.01;                 % rad -- nguong coi la da phuc hoi

P0 = make_P_v13('fis_rwip_v13_best.fis');
assert(isfile(P0.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P0.fisFile);
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

nK = numel(kicks); nC = numel(theta0_list); nM = numel(modes);
peakDev = nan(nK,nC,nM);   % do lech dinh so voi theta ngay truoc khi kich
tRec    = nan(nK,nC,nM);   % thoi gian phuc hoi ke tu luc kich
okRec   = false(nK,nC,nM);

fprintf('Quet %d bien do nhieu x %d dieu kien dau x %d bo dieu khien = %d mo phong\n\n', ...
        nK, nC, nM, nK*nC*nM);

for a = 1:nK
    fprintf('===== Xung = %.2f rad/s tai t = %.1f s =====\n', kicks(a), t_dist);
    fprintf('%-22s', 'theta0 ->');
    for c = 1:nC, fprintf('%12.2f', theta0_list(c)); end
    fprintf('\n');

    for m = 1:nM
        P = P0;
        if strcmpi(modes{m},'SMCESO')
            P.Kp_phi_mag = P0.SMC_Kp_phi_mag;  P.Kv_phi_mag = P0.SMC_Kv_phi_mag;
        elseif strcmpi(modes{m},'PSOFLC_RL')
            P = rl_policy_from_agent(P, RLBC_FILE);   % vong ngoai CHUNG voi PSOFLC thuan
        end
        P.dist = struct('t', t_dist, 'dthetadot', kicks(a));

        fprintf('%-22s', names{m});
        for c = 1:nC
            o = sim_rwip_v13(modes{m}, P, [theta0_list(c);0;0;0], Tf, Ts);
            if o.diverged
                fprintf('%12s','MAT ON DINH'); continue
            end
            iK = find(o.t >= t_dist, 1, 'first');
            th0k = o.theta(max(iK-1,1));               % theta ngay truoc khi kich
            seg  = iK:numel(o.t);
            peakDev(a,c,m) = max(abs(o.theta(seg) - th0k));
            k = find(abs(o.theta(seg)) > recThresh, 1, 'last');
            if isempty(k)
                tRec(a,c,m) = 0;  okRec(a,c,m) = true;
            elseif k < numel(seg)
                tRec(a,c,m) = o.t(seg(k)) - t_dist;  okRec(a,c,m) = true;
            else
                tRec(a,c,m) = inf; okRec(a,c,m) = false;   % chua phuc hoi trong Tf
            end
            if okRec(a,c,m), fprintf('%12.2f', tRec(a,c,m));
            else,            fprintf('%12s','KHONG HOI'); end
        end
        fprintf('\n');
    end
    fprintf('(moi o: thoi gian phuc hoi tinh tu luc kich, giay)\n\n');
end

%% ---------------- Tong hop ----------------
fprintf('===== TONG HOP TREN TOAN DAI DIEU KIEN DAU =====\n');
fprintf('%-22s%10s%14s%14s%14s\n','','xung','ty le phuc hoi','t_rec trung binh','do lech dinh max');
for a = 1:nK
    for m = 1:nM
        r  = okRec(a,:,m);
        tr = tRec(a,:,m);  tr = tr(isfinite(tr));
        pd = peakDev(a,:,m); pd = pd(isfinite(pd));
        fprintf('%-22s%10.2f%12d/%d%14s%14.4f\n', names{m}, kicks(a), sum(r), nC, ...
                ternD(isempty(tr),'--',sprintf('%.2f s',mean(tr))), ...
                ternN(isempty(pd),NaN,max(pd)));
    end
    fprintf('\n');
end

% --- Cong muc 6, phan "thoi gian phuc hoi tai xung 2.0 rad/s" -----------
% Moc dung la "bo lai SMC-ESO" (PSOFLC_SMCESO, 6.17 s trong ban chinh thuc) --
% KHONG PHAI SMCESO doc lap (do la baseline yeu hon, biet truoc, dung no lam
% moc se cho "DAT" gia -- xem ghi chu tuong tu trong Compare_v13_robust.m).
% Tra theo TEN, khong dung so cung, va dung trAll cua LAN CHAY NAY.
iRL     = find(strcmp(modes,'PSOFLC_RL'), 1);
iHybrid = find(strcmp(modes,'PSOFLC_SMCESO'), 1);
a2      = find(kicks == 2.0, 1);
if ~isempty(iRL) && ~isempty(iHybrid) && ~isempty(a2)
    fprintf('Cong muc 6 (thoi gian phuc hoi trung binh tai xung 2.0 rad/s):\n');
    trAll = nan(1,nM);
    for m = 1:nM
        tr = tRec(a2,:,m);  tr = tr(isfinite(tr));
        if ~isempty(tr), trAll(m) = mean(tr); end
        fprintf('  %-22s %s\n', names{m}, ternD(isnan(trAll(m)),'khong phuc hoi du 5/5 IC', ...
                sprintf('%.2f s', trAll(m))));
    end
    if isfinite(trAll(iRL)) && trAll(iRL) <= trAll(iHybrid)
        fprintf('  => DAT moc nay (<= %.2f s cua %s).\n', trAll(iHybrid), names{iHybrid});
    else
        fprintf(2, '  => KHONG DAT moc nay (moc %s = %.2f s).\n', names{iHybrid}, trAll(iHybrid));
    end
end

save('Test_Disturbance_v13_results.mat', ...
     'peakDev','tRec','okRec','kicks','theta0_list','names','modes','P0','t_dist');
fprintf('Da luu Test_Disturbance_v13_results.mat\n');

%% ---------------- Do thi ----------------
figure('Name','Phuc hoi sau nhieu ngoai','Position',[80 80 1050 340*nK]);
for a = 1:nK
    subplot(nK,1,a); hold on; grid on
    P = P0; P.dist = struct('t',t_dist,'dthetadot',kicks(a));
    for m = 1:nM
        Pm = P;
        if strcmpi(modes{m},'SMCESO')
            Pm.Kp_phi_mag = P0.SMC_Kp_phi_mag; Pm.Kv_phi_mag = P0.SMC_Kv_phi_mag;
        elseif strcmpi(modes{m},'PSOFLC_RL')
            Pm = rl_policy_from_agent(Pm, RLBC_FILE);
        end
        o = sim_rwip_v13(modes{m}, Pm, [0.10;0;0;0], Tf, Ts);
        plot(o.t, o.theta, 'LineWidth', 1.3);
    end
    xline(t_dist,'k:'); yline(0,'k:');
    xlim([t_dist-1 min(Tf, t_dist+25)]);
    xlabel('t (s)'); ylabel('\theta (rad)');
    title(sprintf('Xung %.2f rad/s tai t = %.1f s (theta_0 = 0.10)', kicks(a), t_dist));
    if a==1, legend(names,'Location','best'); end
end

function y = ternD(c,a,b), if c, y=a; else, y=b; end, end
function y = ternN(c,a,b), if c, y=a; else, y=b; end, end
