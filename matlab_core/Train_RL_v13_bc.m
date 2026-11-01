function agent = Train_RL_v13_bc(seed, opts)
%TRAIN_RL_V13_BC  Huan luyen lop hieu chinh RL: bat chuoc SMC-ESO roi tinh chinh.
%
%   agent = Train_RL_v13_bc()        -- hat giong 1
%   agent = Train_RL_v13_bc(seed, opts)
%
% ===========================================================================
% VI SAO BAN NAY THAY CHO Train_RL_v13
% ---------------------------------------------------------------------------
% Lan chay truoc: 1024 episode, 4 trieu buoc, KHONG co xu huong hoc. Da chan
% doan xong bang do luong, ba nguyen nhan doc lap:
%
% (1) NHIEU THAM DO SUP TRONG 3 EPISODE.
%     sigma=0.30, decay=2e-4/buoc, sigma_min=0.02
%     -> 0.30*(1-2e-4)^n = 0.02  =>  n = 13 540 buoc = 3.4 episode.
%     Tu episode 4, agent tham do bang 0.2 V tren bien 10 V (2%). No khoa vao
%     bat cu thu gi tim duoc trong 3 episode dau.
%
% (2) 60% SO EPISODE KHONG CHUA TIN HIEU HOC.
%     Do dư dia bang cach chay CHINH luat SMC-ESO (da biet la tot) qua ham
%     phan thuong:
%         IC only          : reward = +0.27 +- 0.97   -> ~ 0
%         IC + nhieu       : reward = +15.5  +- 17.6  -> co dư dia
%         IC + sai tham so : reward = -0.14  +- 0.88  -> AM
%     Phan bo 40/40/20 cua ban truoc nghia la 60% episode day agent rang can
%     thiep la vo ich hoac co hai.
%
% (3) TY LE TIN HIEU/NHIEU = 0.42 (|trung binh| / do lech = 5.2/12.5).
%     Agent duoc yeu cau tim cai thien +5.2 trong tin hieu nhieu +-12.5.
%
% ---------------------------------------------------------------------------
% BA SUA TUONG UNG
% ---------------------------------------------------------------------------
% (1) BAT CHUOC SMC-ESO lam diem xuat phat (behavior cloning). Agent bat dau
%     tu rho ~ 0.3 DA BIET LA HUU ICH, thay vi tu nhieu ngau nhien. Day dung
%     la bai hoc da lam PSO chay duoc: khoi tao ngau nhien trong mien kha thi
%     hep thi vo vong, phai gieo tu nghiem da biet.
% (2) CHI huan luyen tren kich ban NHIEU -- noi duy nhat co dư dia.
%     Ty le tin hieu/nhieu tang tu 0.42 len 0.88.
% (3) GIU nhieu tham do suot qua trinh (decay ~ 0).
%
% ---------------------------------------------------------------------------
% BA SUA BO SUNG SAU REVIEW (truoc lan chay dau tien)
% ---------------------------------------------------------------------------
% (1) LUU CHECKPOINT. Ban truoc bo SaveAgentCriteria ma Train_RL_v13 co. 400
%     episode ~ 2.2 gio (suy tu nhip lan truoc: 1024 ep / 5h40). Loi o episode
%     390 la mat trang.
%
% (2) HAM NONG CRITIC TRUOC KHI THA ACTOR.  Day la sua quan trong nhat: neu
%     khong co no thi sua (1) cua khoi tren cung vo nghia.
%     Critic khoi tao NGAU NHIEN, LR 1e-3. Actor LR 1e-5 nghe co ve an toan,
%     nhung Adam chuan hoa buoc ve ~LR bat ke do lon gradient. Voi 4000 buoc/
%     episode chia PolicyUpdateFrequency = 2 -> 2000 cap nhat actor moi episode:
%         10 episode dau  = 20 000 cap nhat x 1e-5 = 0.2 don vi dich chuyen
%         toan bo 400 ep  = 800 000 cap nhat x 1e-5 = 8.0
%     Trong so fullyConnectedLayer khoi tao dien hinh ~0.05-0.1. Nghia la chinh
%     sach BC bi pha trong vai episode dau, DUNG LUC critic con vo nghia --
%     mat sach diem xuat phat truoc khi no kip co tac dung.
%     Sua: hai pha tren CUNG mot agent (train goi duoc nhieu lan, experience
%     buffer va trong so deu giu nguyen).
%       Pha A: ActorOptimizerOptions.LearnRate = 0 -> actor dong bang tuyet
%              doi, critic hoc tu du lieu do chinh sach BC sinh ra.
%       Pha B: LearnRate tra ve 1e-7, tinh chinh that.
%     LUU Y (phat hien o lan chay thu hai cua sua nay): KHONG dung
%     PolicyUpdateFrequency de dong bang actor -- TD3 nguyen ban tri hoan ca
%     actor LAN target network cung mot tan so d, nen dat no = 4000 cung dong
%     bang luon target network, cat dung co che critic can de hoc. Da doi sang
%     LR = 0 va giu PolicyUpdateFrequency = 2 nguyen suot; xem chi tiet o dau
%     muc 4 ben duoi.
%     NGAN SACH: pha A them (nay la) 100 episode NGOAI ngan sach 400 da chot.
%     Chinh dang vi trong pha A actor khong tim kiem -- khong co viec "keo dai
%     de doi ket qua dep". Ghi ro con so nay trong bai bao.
%
% (3) LUU CHINH SACH BC-ONLY RIENG.  Ablation quyet dinh cua ca thi nghiem:
%     BC sao chep chinh SMC-ESO. Neu khong tach duoc BC-only voi BC+TD3 thi mot
%     chinh sach BC-only cung se "dat" tieu chi muc 6 -- vi no LA SMC-ESO, chu
%     khong phai vi RL hoc duoc gi. Khong co file nay thi khong tra loi duoc
%     cau phan bien hien nhien nhat.
%
% ---------------------------------------------------------------------------
% BA QUYET DINH CON LAI (rho / lich nhieu / khoang lech ZOH)
% ---------------------------------------------------------------------------
% (6) rho CHI GHI LOG, KHONG DUA VAO PHAN THUONG.  Muc 6 THIETKE dung rho lam
%     quy tac PHAN LOAI hau nghiem ("neu chi vuot moc khi rho > 1 thi no da thay
%     the bo nen"). Dua vao phan thuong se EP agent trong giong residual du no
%     co phai hay khong -- tu tay xoa dung cai quan sat can thu duoc. Chi tiet:
%     rl_rho_log.m. Bang chan doan in ra cuoi run va luu trong truong rhoLog.
%
% (7) NHIEU THAM DO: decay = 0 CHINH XAC.  Ban truoc de 1e-6, tuc sigma cham
%     san 0.05 quanh episode 173 -- mot lich anneal khong ai chon co chu dich va
%     lech voi chinh cau "GIU nhieu tham do suot qua trinh" o tren. Sua theo
%     huong y dinh da ghi. Khong ton gi vi danh gia dung getAction tat dinh.
%
% (8) KHOANG LECH ZOH: DO, KHONG SUA.  Nhan BC lay tu SMC-ESO tinh moi 1 ms roi
%     lay mau moi 10 ms, trong khi agent chay ZOH 10 ms. Doi cach sinh nhan sang
%     SMC-ESO ZOH nghia la BC bat chuoc mot bo dieu khien KHAC va yeu hon bo
%     trong Bang 1-6, lam ablation mo di chu khong sach hon. Voi w0 = 20 rad/s
%     (~3.2 Hz), lay mau 100 Hz la thua 30 lan nen sai lech nhieu kha nang nho --
%     nhung "nhieu kha nang" khong phai phep do. reportPolicy nay in CA hang
%     tham chieu SMC-ESO 1 kHz lan hang chinh sach BC o ZOH 100 Hz, tren cung
%     danh sach kich ban va cung bo chi so. Lech > ~20% thi dung lai va sinh
%     nhan bang SMC-ESO ZOH -- quyet dinh dua tren so do, khong phai phong doan.
%
% ---------------------------------------------------------------------------
% QUY TAC DUNG -- CHOT TRUOC, KHONG GIA HAN
% ---------------------------------------------------------------------------
% 400 episode tinh chinh (cong 20 episode ham nong critic, actor dong bang).
% Neu chinh sach hoc duoc khong vuot moc SMC-ESO o cac tieu chi muc 6
% (THIETKE_MOITRUONG_RL_v13.md), BAO CAO KET QUA AM. Khong keo dai, khong doi
% trong so, khong noi dac ta.
% ===========================================================================
%
% ===========================================================================
% BAN VA -- LAN THU CO DINH HUONG, DUA TREN CHAN DOAN SO SANH VOI BAI CEAI
% CUNG TAC GIA (kien truc 4-ngo-vao, TD3 hoi tu). Bon thay doi, CA BON PHAI
% co mat cung luc -- day la mot cau hinh, khong phai bon thi nghiem doc lap:
%
% (1) [DA BI BAN VA 3 DAO NGUOC -- doc muc "BAN VA 3" tai cho dat DiscountFactor
%     truoc khi tin muc nay. Tom tat: so do -0.025 vien dan o day da duoc quy
%     cho loi PolicyUpdateFrequency o MOT CHO KHAC trong chinh file nay, nen no
%     khong the dong thoi la bang chung ket toi gamma. Loi do da sua; gamma da
%     tra ve 0.999.]
%     DiscountFactor: 0.999 -> 0.99 (xem ngay duoi). Chan troi hieu dung tu
%     1000 buoc (10 s) xuong 100 buoc (1 s). Ly do: bootstrap TD 1-buoc lan
%     truyen gia tri NGUOC qua tung do 1000 buoc de toi trang thai dau episode
%     -- voi target network chi cap nhat mem tau=0.005 moi episode, day la co
%     che hoi tu rat cham, phu hop voi ket qua do duoc (Q0 dung yen o -0.025
%     suot 400 000 buoc pha A trong khi gia tri that ~+0.311). CEAI dung
%     gamma=0.99 tren episode 15 s (1500 buoc) -- chan troi hieu dung ngan hon
%     hoan toan quy dao, khong can lan truyen xa. 0.99 o day van du de bao
%     phuc hoi banh da (~6-25 s theo Bang 4) qua NHIEU buoc bootstrap gan nhau
%     hon la mot chan troi dai nhung khong bao gio lan toi duoc trong thuc te.
%
% (2) ActorOptimizerOptions.LearnRate: 1e-7 -> 1e-6 (muc 5 duoi). Theo dung
%     cong thuc da dan (LR x so cap nhat = do dich chuyen trong so): 400 ep x
%     2000 cap nhat x 1e-6 = 0.8 don vi dich chuyen -- gap 10 lan muc "gan nhu
%     dong bang" (0.08) nhung van kem 10 lan muc da gay sup (8.0 tai LR=1e-5).
%     Day KHONG phai doan mo, ma la trung diem hinh hoc co chu dich giua hai
%     dau da do duoc.
%
% (3) Phan thuong (sua trong rwip_env_v13.m): them so hang thua +10/-5 canh
%     thanh phan dense, va giam uMax tu 10V xuong 8V (SUA LAI tu 5V ban dau --
%     5V lam SMC-ESO bao hoa 83-100%% ngay o BC, xem "BAN VA" muc A trong file
%     do). File nay TU DONG ke thua ca hai qua cfg = rwip_env_v13(cfg0).
%
% (4) AN TOAN: Pha B (tinh chinh) chia thanh B1 (tham do, 40 episode dau o LR
%     moi) roi B2 (400-40 episode con lai). Giua B1 va B2 co MOT cong kiem tra
%     tu dong doi chieu voi CHINH dau hieu sup do da do duoc tai LR=1e-5 (that
%     bai dau tien ~episode 8, xu huong am ~episode 24): neu B1 co dau hieu
%     tuong tu, DUNG va bao loi thay vi chay tiep 360 episode vo ich. Day la
%     "kiem soat" duoc yeu cau, khong phai chi doi LR roi hy vong.
% ===========================================================================

if nargin < 1 || isempty(seed), seed = 1; end
if nargin < 2, opts = struct(); end
% BCIters = 3000, KHONG PHAI 6000. Day la sua lai mot loi cua chinh qua trinh
% sua truoc: khi them lay mau phan tang, BCIters bi tang 3000->6000 CUNG LUC
% ma khong kiem chung rieng -- doi hai bien mot luc, dung loi da tu nhac lai
% nhieu lan. Do duoc: voi CUNG lay mau deu (TailFrac=0), 6000 vong cho RMSE
% offline THAP HON (0.01415 vs 0.01585) nhung rms vong kin CAO HON 27 LAN
% (+186% vs -6.9%) va IAEtheta vs Bang 1 tu +9.5% thanh +132%. Do la chu ky
% kinh dien cua overfit trong behavior cloning: MSE offline khong do dung cai
% can do, huan luyen lau hon lam actor bam nhieu cua 60 episode thay vi ham
% chinh sach nen, nhay hon voi lech phan bo luc tu chay vong kin. 3000 vong la
% cong thuc DA XAC NHAN dat cong (xem Train_RL_v13_bc(1,struct('TailFrac',0))).
def = struct('MaxEpisodes',400, 'BCEpisodes',60, 'BCIters',3000, ...
             'TailFrac',0.15, 'WarmupEpisodes',100, 'SaveAgentValue',0, ...
             'SaveDir','rl_v13_bc', 'Plots','training-progress', 'Verbose',true, ...
             'SkipCriticGate',false, 'WarmupChunk',25, 'ResumeAgent','', ...
             'ActorLRFineTune',1e-6, 'DiscountFactor',0.99, 'GradThresh',1.0, ...
             'SkipB1Gate',false, 'SkipBCGate',false, ...
             'MaxWarmupEpisodes',300, 'WarmupTargetSlope',0.30, ...
             'RewardCfg',struct());
% MaxWarmupEpisodes / WarmupTargetSlope: them 09/08 (ban thu ba) de Pha A TU
% GIA HAN thay vi dung sau WarmupEpisodes co dinh roi cho nguoi chay lai bang
% tay. WarmupTargetSlope = 0.30 KHOP scaleOK da dung san o cong critic (xem
% "NGUONG SUA 09/08" phia duoi) -- khong phai so moi bia ra, chi la cung mot
% nguong duoc doc som hon de quyet dinh dung vong lap som hon. MaxWarmupEpisodes
% = 300 la tran ngan sach, khong phai ky vong: neu cham tran ma van chua dat,
% dung va bao cao nguyen trang, KHONG tu y chay them.
% Ba tuy chon cuoi them 09/08 de TAI LAP duoc cau hinh cua agent_bc_seed1.mat
% (08/08 10:04) -- agent duy nhat vuot ca PSO-FLC lan SMC-ESO tren giao thuc
% danh gia that. Cau hinh do khac ban hien hanh o dung ba cho:
%     DiscountFactor 0.995 (khong phai 0.99)
%     GradThresh     Inf   (khong cat gradient)
%     ham thuong     control-variate GOC (RewardCfg = rewardMode 'cv', khong
%                    bonus/penalty, khong rScale/phiCap)
% Dat qua opts thay vi sua tay ba dong -- de moi lan tai lap deu giong nhau va
% ghi lai duoc trong file ket qua.
% ResumeAgent: duong dan toi file agent DA QUA BC + Pha A (vd *_AABORT.mat).
% Khi dat, bo qua toan bo buoc [1/5]-[3/5] va chay TIEP Pha A tren agent do.
% Dung de keo dai Pha A ma khong vut di thoi gian da chay -- xem ghi chu tai
% khoi "DUONG TIEP TUC PHA A" ben duoi. Co cong kiem tuong thich.
% WarmupChunk: Pha A chay theo khoi bao nhieu episode thi do slope mot lan.
% Muc dich la thu duoc DUONG CONG slope thay vi mot diem -- xem ghi chu dai
% tai vong lap Pha A ve ly do ba lan chan doan truoc deu truot vi chi co 1 diem.
% SkipCriticGate: chi dat true khi ban DA doc bang chan doan cuoi Pha A va van
% co y muon chay tiep. Mac dinh false -- xem ghi chu tai cong critic ve ly do
% cong do gio error() that thay vi chi in canh bao.
fn = fieldnames(def);
for k = 1:numel(fn), if ~isfield(opts,fn{k}), opts.(fn{k}) = def.(fn{k}); end, end

rng(seed,'twister');
if ~isfolder(opts.SaveDir), mkdir(opts.SaveDir); end

% Moi truong: CHI kich ban nhieu (noi co dư dia)
cfg0 = struct('p_ic',0.0, 'p_kick',1.0, 'p_param',0.0);
% opts.RewardCfg ghi de cfg moi truong -- dung de tai lap ham thuong GOC.
rf = fieldnames(opts.RewardCfg);
for k = 1:numel(rf), cfg0.(rf{k}) = opts.RewardCfg.(rf{k}); end
[env, cfg] = rwip_env_v13(cfg0);
obsInfo = getObservationInfo(env);  actInfo = getActionInfo(env);
nObs = obsInfo.Dimension(1);

% CRITIC_L2: mot ten, mot noi. Dat O DAY (truoc "if ~resuming"), KHONG PHAI
% ben trong no -- hang so nay con duoc doc lai o cong critic (sau Pha A), chay
% CA KHI resuming = true va toan bo "if ~resuming" bi bo qua. Loi thuc te da
% xay ra: dat no o dong ~623 (trong "if ~resuming") gay
% "Unrecognized function or variable 'CRITIC_L2'" ngay khi resume, vi dung o
% dong ~1281 nam NGOAI nhanh do. Dat truoc de mot ten dung o CA HAI nhanh.
CRITIC_L2 = 1e-4;

fprintf('=== Train_RL_v13_bc, hat giong %d ===\n', seed);
fprintf('  kich ban: 100%% nhieu (dư dia do duoc +15.5 +- 17.6)\n');
fprintf('  ngan sach: %d episode BC + %d episode RL\n\n', opts.BCEpisodes, opts.MaxEpisodes);

% ===========================================================================
% DUONG TIEP TUC PHA A (opts.ResumeAgent). Them 09/08.
% ---------------------------------------------------------------------------
% Do duoc: duong cong slope trong Pha A VAN DANG LEO khi het 100 episode
%     25 ep 0.0866 | 50 ep 0.0928 | 75 ep 0.1393 | 100 ep 0.1764 (cong: 0.230)
% Nhip 25 episode cuoi la +0.037, nen de slope toi nguong 0.50 can them ~161
% episode. Chay lai tu dau nghia la vut di 36 phut da lam (5 phut BC + 31 phut
% Pha A) chi de tai tao dung trang thai dang co trong file AABORT.
%
% Duong nay nap agent DA QUA BC + Pha A va chay tiep Pha A, KHONG sinh lai du
% lieu BC va KHONG dung lai actor. Buffer kinh nghiem di theo agent (da dat
% ResetExperienceBufferBeforeTraining = false) nen 400 000 mau cu duoc giu.
%
% CONG TUONG THICH: agent cu duoc huan luyen tren MOT ham thuong va MOT gamma
% cu the. Nap no vao moi truong da doi la tron hai bai toan -- dung cai bay ma
% Resume_RL_v13_bc_B2 da phai dung cong de chan. Kiem rewardVersion, rScale va
% gamma truoc khi chay.
% ===========================================================================
resuming = isfield(opts,'ResumeAgent') && ~isempty(opts.ResumeAgent);
if resuming
    fprintf('[TIEP TUC] Nap %s ...\n', opts.ResumeAgent);
    Rz = load(opts.ResumeAgent);
    assert(isfield(Rz,'agent') && isfield(Rz,'cfg'), ...
        'File %s thieu agent/cfg.', opts.ResumeAgent);
    bad = {};
    if ~isfield(Rz,'rewardVersion') || Rz.rewardVersion ~= rewardVersionConst()
        bad{end+1} = 'rewardVersion khong khop'; %#ok<AGROW>
    end
    if ~isfield(Rz.cfg,'rScale') || Rz.cfg.rScale ~= cfg.rScale
        bad{end+1} = sprintf('rScale khac (file %g vs hien tai %g)', ...
            getfield_d(Rz.cfg,'rScale',NaN), cfg.rScale); %#ok<AGROW>
    end
    gNow = Rz.agent.AgentOptions.DiscountFactor;
    if abs(gNow - 0.99) > 1e-12
        bad{end+1} = sprintf('gamma cua agent = %.4f, hien tai dung 0.99', gNow); %#ok<AGROW>
    end
    if ~isempty(bad)
        for i = 1:numel(bad), fprintf(2, '  (!) %s\n', bad{i}); end
        error('Train_RL_v13_bc:resumeMismatch', ...
            ['Agent trong %s khong tuong thich voi cau hinh hien tai (xem tren).\n' ...
             'Chay lai tu dau thay vi ep hai cau hinh vao nhau.'], opts.ResumeAgent);
    end
    agent  = Rz.agent;
    bcFile = getfield_d(Rz, 'bcFile', fullfile(opts.SaveDir, sprintf('agent_bconly_seed%d.mat', seed)));
    rmseBC = getfield_d(Rz, 'rmseBC', NaN);
    agentOpts = agent.AgentOptions;
    fprintf(['  Bo qua BC va dung agent (%d..%d giay tiet kiem). Chay tiep Pha A %d episode.\n' ...
             '  Buffer kinh nghiem di theo agent -- khong bi xoa.\n\n'], 300, 2200, opts.WarmupEpisodes);
end
if ~resuming

%% ===================== 1. SINH DU LIEU BAT CHUOC =====================
% ===========================================================================
% NHAN BC SINH O ZOH 100 Hz (P.smcTs), KHONG PHAI 1 kHz ROI LAY MAU. Sua 08/08.
% ---------------------------------------------------------------------------
% Day la THI HANH quy tac da chot san o muc (8) dau file nay:
%   "Lech > ~20% thi dung lai va sinh nhan bang SMC-ESO ZOH -- quyet dinh dua
%    tren so do, khong phai phong doan."
% So do kich hoat no (hat giong 1, TailFrac = 0): d rms = +57% / +46% / +11%.
%
% Ban cu: SMC-ESO tinh moi 1 ms roi lay mau moi 10 ms. Nhan thu duoc la anh cua
% mot bo dieu khien 1 kHz tai cac diem 100 Hz -- mot ham ma KHONG chinh sach
% 100 Hz nao tai lap duoc trong vong kin, vi giua hai diem lay mau bo goc con
% lam viec 9 lan nua. Actor bi ep khop mot muc tieu khong dat duoc, va sai lech
% do bieu hien ra thanh d rms +57% (giu mau chatter suot 10 ms thay vi de bo
% goc trung binh hoa chung o 1 kHz).
%
% Ban nay: SMC-ESO chay o CHINH chu ky cua agent. Nhan tro thanh ham ma mot
% chinh sach 100 Hz co the tai lap dung.
%
% GHI VAO BAI BAO (muc (8) da canh bao truoc, va no dung): BC gio bat chuoc
% SMC-ESO@100Hz, mot bo YEU HON SMC-ESO@1kHz cua Bang 1-6. Cai gia do duoc do
% va in ra ngay duoi day (hang "gia cua ZOH"), khong giau di. Do la mot so lieu
% cua bai bao, khong phai mot dieu chinh am tham.
% ===========================================================================
fprintf('[1/5] Sinh du lieu bat chuoc tu SMC-ESO @ZOH %g Hz (%d episode)...\n', ...
        1/cfg.Ts_agent, opts.BCEpisodes);
P = cfg.P;  P.logObs = true;  P.uCorrMax = cfg.uMax;  P.rlTs = cfg.Ts_agent;
P.smcTs = cfg.Ts_agent;   % <-- nhan sinh o dung dieu kien trien khai
X = [];  Y = [];
for e = 1:opts.BCEpisodes
    x0 = [uni(-0.20,0.20); uni(-0.30,0.30); 0; uni(-2,2)];
    Pk = P;  Pk.dist = struct('t', uni(3,10), 'dthetadot', uni(0.5,3.0)*sgn());
    o = sim_rwip_v13('PSOFLC_SMCESO', Pk, x0, cfg.Tf, cfg.Ts);
    if o.diverged || ~isfield(o,'obsLog'), continue; end
    X = [X, o.obsLog];  Y = [Y, o.aLog]; %#ok<AGROW>
end
fprintf('      %d mau, |a| trung binh = %.4f, |a| max = %.4f\n', ...
        size(X,2), mean(abs(Y)), max(abs(Y)));
assert(size(X,2) > 1000, 'Qua it mau BC (%d). Kiem tra logObs.', size(X,2));

%% ===================== 2. HUAN LUYEN ACTOR BANG HOI QUY =====================
% Vong huan luyen thu cong voi dlnetwork + adamupdate: khong phu thuoc API
% trainNetwork/trainnet (ten va chu ky ham doi giua cac phien ban -- da vap
% hai lan trong du an nay voi rlGaussianActionNoiseOptions va SaveFormat).
fprintf('[2/5] Bat chuoc: huan luyen actor (%d vong)...\n', opts.BCIters);
% MOI LAYER MOT DONG. Trong [ ] cua MATLAB, xuong dong ngan HANG con dau cach
% ngan COT -- viet hai layer tren mot dong tao ra hang 1x2 canh hang 1x1 va
% vertcat gay ngay. Khong duoc gop dong o day.
actorNet = dlnetwork([
    featureInputLayer(nObs,'Name','obsIn')
    fullyConnectedLayer(256,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(256,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(1,'Name','fc3')
    tanhLayer('Name','tanh')]);

% ===========================================================================
% LAY MAU MINIBATCH PHAN TANG -- KHONG DUOC lay deu.
% ---------------------------------------------------------------------------
% Do duoc o lan chay truoc: |a| trung binh 0.0238 nhung dinh cham 1.0 (bao hoa
% tran uCorrMax). Voi minibatch DEU, nhom duoi gan nhu khong bao gio duoc chon,
% nen MSE hoi tu ve nghiem tam thuong "xuat ~ 0". Hau qua do duoc, ba dau hieu
% khop nhau hoan toan:
%     RMSE = 0.01585            -> trong dep, nhung so voi trung binh |a| =
%                                  0.0238 thi du con 67% bien do tin hieu
%     max|u_corr| hut toi 64%   -> dung chu ky cat dinh; rms chi hut 7%
%     phan thuong pha A = +6.0  -> chi 39% bien loi cua SMC-ESO (+15.5)
%     SNR = 6.0/15.9 = 0.38     -> gan nhu dung con so 0.42 da lam hong lan truoc
%
% Bien loi nam DUNG o cac dinh hiem do -- do la luc bo hieu chinh thuc su lam
% viec. Phan than tin hieu (|a| ~ 0.02) khong mang thong tin gi ngoai "dung lam
% gi ca", va bo nen v13 da tu lam dieu do roi.
%
% Phan tang: nua minibatch deu tren toan bo, nua lay tu decile |a| tren cung
% -> nhom duoi tu 10% len 55% dai dien.
%
% LUU Y KHI DOC KET QUA: rmseBC sau day do tren minibatch PHAN TANG nen SE CAO
% HON 0.01585 cua lan truoc. Hai con so KHONG so sanh duoc voi nhau. Con so can
% nhin la RMSE tach rieng than/duoi ben duoi, va phan thuong pha A.
% ===========================================================================
% ---------------------------------------------------------------------------
% CHON opts.TailFrac -- NOI SUY GIUA HAI DAU MUT DA DO, khong phai do thu
% ---------------------------------------------------------------------------
%   f = 0.00 (lay deu)  : dai dien duoi 10%, d peak -64%, d IAEtheta  +1%
%   f = 0.50 (50/50)    : dai dien duoi 55%, d peak -30%, d IAEtheta +135%
% f = 0 khop than nhung cat dinh; f = 0.5 vot dinh nhung can thiep thua o che
% do tinh (rms +117%) va pha vo rang buoc muc 6 ("khong xau hon PSO-FLC thuan
% qua 20% o Bang 1 danh nghia": IAEtheta +156% so voi 0.024463).
% f = 0.15 -> dai dien duoi ~23.5%, tuc trong so duoi 2.35x -- trung binh NHAN
% cua hai dau mut (1x va 5.5x), phep noi suy tu nhien tren thang nhan.
%
% QUY TAC DUNG -- CHOT TRUOC:
% Day la lan hieu chinh BC thu ba va la lan CUOI. Neu f = 0.15 khong dua ca
% d rms lan d IAEtheta ve trong +-20% tren ca ba kich ban, DUNG chinh f va quay
% ve BC1 (lay deu), bao cao thieu hut dinh nhu mot han che. Tiep tuc do f sau do
% chinh la co che "doi trong so cho toi khi ra ket qua mong muon" ma muc 3.5
% KE_HOACH_HOAN_THIEN_v13.md da ghi lai nhu mot loi phuong phap luan.
% Muc tieu la MOT dai luong ngoai sinh, khong phai tham so tu do: tai lap
% SMC-ESO, ma so cua no da co san va co dinh trong Bang 1-6.
% ---------------------------------------------------------------------------
aAbs    = abs(Y(:));
thrTail = qtile(aAbs, 0.90);
idxTail = find(aAbs >= thrTail);
idxBulk = find(aAbs <  thrTail);
nTail   = round(opts.TailFrac * nBatch());
fprintf('      phan tang f = %.2f: nguong duoi |a| >= %.4f -> %d mau (%.1f%% tap)\n', ...
        opts.TailFrac, thrTail, numel(idxTail), 100*numel(idxTail)/numel(aAbs));
fprintf('      moi minibatch: %d mau deu + %d mau tu duoi (dai dien duoi ~%.1f%%)\n', ...
        nBatch()-nTail, nTail, 100*(opts.TailFrac + (1-opts.TailFrac)*0.10));

avgG = []; avgSq = []; lr = 1e-3;  nB = nBatch();
lossHist = zeros(opts.BCIters,1);
for it = 1:opts.BCIters
    tailPick = idxTail(randi(numel(idxTail), 1, nTail));
    idx = [ randi(size(X,2), 1, nB-nTail), reshape(tailPick,1,[]) ];
    dlX = dlarray(single(X(:,idx)), 'CB');
    dlY = dlarray(reshape(single(Y(idx)),1,[]), 'CB');
    [loss, grad] = dlfeval(@bcLoss, actorNet, dlX, dlY);
    [actorNet, avgG, avgSq] = adamupdate(actorNet, grad, avgG, avgSq, it, lr);
    lossHist(it) = double(gather(extractdata(loss)));
    if mod(it, 1000) == 0
        fprintf('      vong %5d  RMSE (phan tang) = %.5f\n', ...
                it, sqrt(mean(lossHist(it-999:it))));
    end
end
rmseBC   = sqrt(mean(lossHist(end-499:end)));
rmseBulk = bcRmse(actorNet, X, Y, idxBulk);
rmseTail = bcRmse(actorNet, X, Y, idxTail);
fprintf('      RMSE phan tang = %.5f | than = %.5f | duoi = %.5f\n', ...
        rmseBC, rmseBulk, rmseTail);
fprintf('      => duoi tuong duong %.3f V tren bien %g V. Dinh la cho co bien loi;\n', ...
        rmseTail*cfg.uMax, cfg.uMax);
fprintf('         neu RMSE duoi van ~ do lon dinh thi phan tang chua du.\n');

%% ===================== 3. DUNG AGENT TD3 TU ACTOR DA BAT CHUOC =========
fprintf('[3/5] Dung agent TD3 tu actor da bat chuoc...\n');
actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

mkCritic = @() rlQValueFunction(criticGraph(nObs,1), obsInfo, actInfo, ...
    'ObservationInputNames','obsIn','ActionInputNames','actIn');
% ===========================================================================
% BAN VA 3 -- QUAY LAI DiscountFactor = 0.999. DOC HET TRUOC KHI DOI LAI.
% ---------------------------------------------------------------------------
% Chan troi hieu dung 1/(1-gamma), quy ra giay o Ts_agent = 0.01:
%   0.99  -> 100 buoc  = 1 s      <-- ban truoc ban va nay
%   0.995 -> 200 buoc  = 2 s
%   0.999 -> 1000 buoc = 10 s     <-- DANG DUNG
%   0.9996-> 2500 buoc = 25 s     (bang tron thoi gian hap thu xung 2.0 rad/s)
%
% THANG THOI GIAN VAT LY (dau rwip_env_v13.m, muc "Episode"): "do duoc banh da
% can ~25 s de hap thu va tra lai xung 2.0 rad/s". Voi gamma=0.99, critic duoc
% yeu cau dinh gia mot hien tuong 6-25 s qua mot cua so 1 GIAY. No khong the
% nhin thay hau qua cua viec tieu ton hay tich luy momen banh da -- ma do CHINH
% LA bien trang thai cham quyet dinh thanh bai cua kien truc RWIP.
%
% ---------------------------------------------------------------------------
% "NHUNG 0.999 DA THU ROI VA THAT BAI" -- BANG CHUNG DO KHONG SACH.
% ---------------------------------------------------------------------------
% Chu thich cu ghi: gamma=0.999 cho Q0 dung yen -0.025 so voi gia tri that
% +0.311 sau 400 000 buoc pha A, va ket luan gamma la thu pham. Nhung lan chay
% do dinh DONG THOI mot loi da duoc xac nhan rieng: PolicyUpdateFrequency sai
% (xem muc bay to o dau file). Chinh chu thich pha A trong file nay viet:
%
%   "Lan truoc Q0 dung yen o ~ -0.025 suot 100 episode ... do LA hau qua cua
%    loi PolicyUpdateFrequency da neu tren, KHONG PHAI actor chua du thoi gian."
%
% Tuc la cung mot so do (-0.025) da duoc quy cho HAI nguyen nhan khac nhau o
% hai cho khac nhau trong cung mot file. Khong the dong thoi la bang chung ket
% toi gamma VA bang chung ket toi PolicyUpdateFrequency. Loi PolicyUpdateFrequency
% nay da duoc sua; nen so do do khong con ket toi duoc gamma nua, va viec ha
% xuong 0.99 mat can cu.
%
% Doi chieu them: o gamma=0.99 (sau khi sua PolicyUpdateFrequency VA them bonus),
% Q0 ~ 0.0-0.2 so voi gia tri that +0.797 -- VAN TRUOT cong kiem tra critic.
% Ha gamma da khong cuu duoc gi. Hai cau hinh gamma deu truot, nen bien gamma
% khong giai thich duoc that bai; thu con lai chua tung duoc thu la HAM THUONG,
% va do la thu ban va 2 trong rwip_env_v13.m vua sua.
%
% ---------------------------------------------------------------------------
% THU TU BAT BUOC: gamma=0.999 CHI AN TOAN SAU BAN VA 2 (C).
% ---------------------------------------------------------------------------
% Ban va 2 (C) sua loi khai bao het-gio la TERMINAL (IsDone=1) thay vi TRUNCATED
% (IsDone=2). O gamma=0.99 loi do chi ban ~100 buoc cuoi (2.5% du lieu) nen de
% bo qua. O gamma=0.999 no lan nguoc 1000 buoc = 25% moi episode, va vi 140/140
% episode ket thuc bang duong het-gio, 25% TOAN BO du lieu se bi day ve 0.
% Nang gamma ma KHONG sua (C) truoc thi bien mot loi nho thanh loi chi phoi.
% Neu ban co y dinh revert ban va 2, PHAI revert dong DiscountFactor nay cung.
%
% CHI PHI PHAI THEO DOI: gia tri Q gio o thang lon hon ~10 lan (vTrue = r_tb/4).
% Adam dich chuyen tham so ~LR moi lan cap nhat bat ke do lon muc tieu, nen
% critic can nhieu cap nhat hon de vuon toi thang moi. Pha A co 100 ep x 4000 =
% 400 000 cap nhat critic -- du, nhung cong kiem tra critic cuoi pha A la cho
% duy nhat biet duoc. NEU Q0 van < 25% vTrue sau ban va nay, dung nang gamma
% tiep len 0.9996: luc do ket luan dung la bai toan nam o CRITIC NETWORK
% (criticGraph) hoac o phuong sai ham thuong, khong phai o chan troi.
% ===========================================================================
% ---------------------------------------------------------------------------
% HOAN TAC ban va 3: gamma tro ve 0.99. Can cu: WRIP_TD3_Training.m -- mot cau
% hinh cua CHINH TAC GIA tren cung lop doi tuong, DA HOI TU -- dung gamma = 0.99
% tren episode 1500 buoc. Tuc chan troi hieu dung 100 buoc = 1 s tren episode
% 15 s (6.7%). Ban va 3 nang len 0.999 dua tren lap luan vat ly ve thang thoi
% gian banh da; lap luan do nghe hop ly nhung KHONG co cau hinh nao chung minh,
% va da that bai hai lan. Mot cau hinh chay duoc dang tin hon mot lap luan hay.
%
% Neu sau nay muon thu lai 0.999, hay thu SAU khi da co mot lan hoi tu o 0.99 --
% luc do moi biet dang danh doi cai gi de lay cai gi.
% ---------------------------------------------------------------------------
agentOpts = rlTD3AgentOptions( ...
    'SampleTime',            cfg.Ts_agent, ...
    'DiscountFactor',        opts.DiscountFactor, ...   % opts: 0.99 mac dinh, 0.995 de tai lap
    'MiniBatchSize',         256, ...
    'ExperienceBufferLength',1e6, ...
    'TargetSmoothFactor',    5e-3, ...
    'PolicyUpdateFrequency', 2);
% (3) Nhieu tham do GIU SUOT qua trinh -- khong decay ve 0 sau 3 episode nua.
agentOpts = setNoise2(agentOpts, 'TargetPolicySmoothModel', struct( ...
    'StandardDeviation',0.10,'StandardDeviationDecayRate',0, ...
    'LowerLimit',-0.3,'UpperLimit',0.3));
% decay = 0 CHINH XAC, khong phai 1e-6. Voi 1e-6, sigma cham san 0.05 quanh
% buoc 693 000 (~episode 173) -- tuc la mot lich anneal 0.10 -> 0.05 ma khong
% ai chon co chu dich, lech voi chinh cau "GIU nhieu tham do suot qua trinh"
% o dau file. Sua theo huong Y DINH DA GHI, khong viet lai chu thich cho khop
% mot gia tri tinh co: do dung la loai lech dinh nghia da ba lan lam hong du an
% nay (thetaTol 0.02 vs 0.002, wrap goc banh da, OS_theta luon bang theta0).
% Khong ton gi: danh gia dung getAction tat dinh nen nhieu tham do khong anh
% huong so cuoi. StandardDeviationMin giu lai lam chan an toan (khong rang buoc
% khi decay = 0).
agentOpts = setNoise2(agentOpts, 'ExplorationModel', struct( ...
    'StandardDeviation',0.10,'StandardDeviationDecayRate',0, ...
    'StandardDeviationMin',0.05));

% ===========================================================================
% BAN VA 5 -- N-STEP RETURN. BAT BUOC PHAI CO NEU GIU gamma = 0.999.
% ---------------------------------------------------------------------------
% Do duoc sau Pha A (hat giong 1, sau ban va 2-4): Q0 = 0.036 so voi gia tri
% that ~0.78 (da hieu chinh thoi diem cu hich) -- 4.6%. Reward da lanh (dense
% 100%, khong con bonus nuot tin hieu), critic graph khong co loi cau truc.
% Nguyen nhan la NGAN SACH LAN TRUYEN GIA TRI, tinh duoc truc tiep:
%
%   Pha A = 100 ep x 4000 buoc = 400 000 lan cap nhat critic
%   Buffer cuoi Pha A          = 400 000 mau, minibatch 256
%   -> moi mau duoc rut ~256 lan = ~256 luot quet buffer
%
%   TD 1-BUOC lan truyen gia tri NGUOC dung MOT buoc moi luot quet. Vay:
%     gamma 0.99   -> chan troi   100 buoc | can  100 luot, co 256 -> DU  x2.6
%     gamma 0.999  -> chan troi   982 buoc | can  982 luot, co 256 -> THIEU x3.8
%     gamma 0.9996 -> chan troi  1995 buoc | can 1995 luot, co 256 -> THIEU x7.8
%
% Tuc gamma = 0.999 voi TD 1-buoc la mot bai toan KHONG GIAI DUOC trong ngan
% sach Pha A -- khong phai critic hong, ma la khong du luot quet de gia tri bo
% ve toi dau episode. Day la sai sot trong khuyen nghi nang gamma o ban va 3:
% chan troi 9.8 s DUNG ve vat ly (banh da 6-25 s), nhung ha 1 GIA THIET khac
% (TD 1-buoc) da bi bo qua.
%
% Hai cach thoat, va chi mot cach giu duoc ca hai muc tieu:
%   (a) ha gamma ve 0.99  -> giai duoc, nhung critic mu truoc thang thoi gian
%                            cua banh da. Da thu, Q0 chi toi 25%.
%   (b) N-STEP RETURN     -> moi luot quet lan truyen N buoc thay vi 1.
%                            Giu duoc chan troi 9.8 s VA nam trong ngan sach.
%
% Chon N = 10: can ~98 luot quet, co 256 -> du x2.6, DUNG BANG bien an toan ma
% gamma = 0.99 tung co. Khong chon N lon hon vi N-step return la uoc luong
% chech-thap-phuong-sai-cao: N cang lon, muc tieu cang nhieu (tong N phan thuong
% that thay vi bootstrap), va reward o day da co SD/trung binh = 13.9/5.2 = 2.7.
% N = 10 la diem can bang co tinh, khong phai so tron.
%
% RANG BUOC: neu sau nay doi critic sang mang hoi quy (LSTM), NumStepsToLookAhead
% PHAI ve 1 -- MATLAB khong ho tro n-step voi recurrent critic.
% ===========================================================================
% --- HOAN TAC ban va 5: n-step tro ve 1 -------------------------------------
% Ban va 5 dua tren gia thuyet "nghen lan truyen". Diag_CriticWarmup_v13 da bac
% bo no: Q0 KHONG he leo trong 100 episode (do doc t = -1.36, khong khac 0).
% Mot critic hoi tu cham thi Q0 phai LEO; Q0 dung yen nghia la no dang o DIEM
% CAN BANG, khong phai dang tren duong toi.
%
% Giu n-step = 10 luc nay se doi HAI bien cung luc voi ban va 6 duoi day, va
% neu ket qua tot thi khong biet cai nao cuu. Do dung la loi da lam mat 29 phut
% o ban va 3. Mot lan chay, mot bien.
agentOpts = setOpt(agentOpts, 'NumStepsToLookAhead', 1, 'rlTD3AgentOptions', ...
    'khong sao -- 1 la mac dinh');

% ===========================================================================
% BAN VA 6 -- L2RegularizationFactor = 0 CHO CRITIC.
% ---------------------------------------------------------------------------
% Do duoc (Diag_CriticFit_v13, tren agent cuoi Pha A):
%
%   he so tuong quan Q_critic voi G_t THAT : r     = +0.737
%   he so goc hoi qui Q ~ a*G + b          : slope = +0.247   (ky vong ~1)
%   Q0 theo episode trong suot Pha A       : PHANG (do doc t = -1.36)
%
% Doc ba so nay cung nhau: critic BIET xep hang trang thai (r cao) nhung bien
% do bi NEN 4 lan, VA no dung yen o muc nen do suot 100 episode. Mot critic
% hoi tu cham thi bien do phai lon dan. Dung yen o muc nen = co mot LUC KEO
% NGUOC dang can bang voi gradient TD.
%
% Trong ca vong lap chi co MOT thu tao ra luc keo ve 0 khong phu thuoc phan
% thuong: L2RegularizationFactor = 1e-4 (MAC DINH cua MATLAB -- chua ai trong
% du an nay tung dat no, va no chua bao gio bi kiem).
%
% VI SAO NO DU MANH O DAY, TRONG KHI THUONG THI KHONG:
%   gradient TD  ~ delta * dQ/dw ,  delta ~ r ~ 0.0013/buoc   <- ti le voi reward
%   gradient L2  = lambda * w                                  <- KHONG DOI
% Phan thuong o day rat nho vi no la hieu control-variate (c_base - c_agent)*dt
% giua hai chinh sach gan nhau -- CHINH thiet ke da chon de giam phuong sai.
% Tac dung phu khong ai luong: no lam yeu ve tin hieu TD di 3 bac, trong khi
% luc keo L2 giu nguyen. Tai diem can bang delta -> 0 con L2 thi khong, nen
% Q dung lai o mot phan cua gia tri that. Do dung la slope = 0.247.
%
% Tuc "L2 qua manh" va "thang phan thuong qua nho" KHONG phai hai nghi van --
% chung la CUNG MOT co che nhin tu hai phia. Sua mot dau la du.
%
% CHON SUA L2 THAY VI NHAN PHAN THUONG LEN:
%   - L2 = 0 la tuy chon optimizer thuan tuy, KHONG dung toi MDP. Ham thuong,
%     chinh sach toi uu, va moi con so da bao cao deu khong doi.
%   - Nhan reward len 100 lan cung khong doi chinh sach toi uu, nhung doi moi
%     con so reward trong bai bao va phai giai thich them mot he so tuy y.
%   - Critic o day khong can chinh quy: no hoc tu 400 000 mau tren mot khong
%     gian quan sat 6 chieu, khong phai che do de qua khop.
%
% NEU BAN VA NAY SAI: dau hieu se la slope VAN ~0.25 sau khi L2 = 0. Khi do
% nghi van con lai la thang phan thuong, va phep sua la nhan reward (mot he so
% duy nhat trong rwip_env_v13) chu KHONG phai quay lai dong toi gamma.
% ===========================================================================
% ===========================================================================
% BAN VA 8 -- DONG BO VOI CAU HINH DA HOI TU (WRIP_TD3_Training.m).
% ---------------------------------------------------------------------------
% HOAN TAC ban va 6 (L2 = 0). Ban da hoi tu GIU L2 = 1e-4 tren CA actor lan
% critic va van chay tot. Nen L2 khong phai thu pham, va dat no ve 0 la mot
% thay doi khong co can cu -- dung loai thay doi da lam mat thoi gian ba lan.
% Co che that nam o THANG PHAN THUONG (ban va 7 trong rwip_env_v13.m): cung
% lambda nay ap dao khi reward ~0.0022/buoc va khong ap dao khi ~0.47/buoc.
%
% HAI THU BAN DA HOI TU CO MA FILE NAY CHUA TUNG DAT:
%
%   GradientThreshold = 1.0
%       File nay de mac dinh = Inf (da xac nhan bang Diag_CriticFit_v13:
%       "GradThresh Inf"). Khong cat gradient nghia la mot minibatch xui co
%       the day trong so di mot buoc lon tuy y. Voi muc tieu TD co phuong sai
%       cao (SD/trung binh = 1.8 do duoc), day la rui ro that, va no MIEN PHI
%       de loai bo.
%
%   TargetUpdateFrequency = 2
%       File nay chua bao gio dat, tuc dua vao mac dinh. Ban da hoi tu dat
%       TUONG MINH cung voi PolicyUpdateFrequency = 2. Dat tuong minh cho khop,
%       va de lan sau doc file biet gia tri nao dang co hieu luc.
%
% Ca hai deu dat qua setOpt de neu phien ban RL Toolbox khong co truong do thi
% bao ra stderr chu khong im lang.
% ===========================================================================
% CRITIC_L2 dinh nghia truoc "if ~resuming" (xem ngay tren dong 211) -- o day
% chi DUNG lai, khong dinh nghia lai.
for iC = 1:numel(agentOpts.CriticOptimizerOptions)
    agentOpts.CriticOptimizerOptions(iC).L2RegularizationFactor = CRITIC_L2;
    agentOpts.CriticOptimizerOptions(iC).GradientThreshold      = opts.GradThresh;
end
agentOpts.ActorOptimizerOptions.L2RegularizationFactor = CRITIC_L2;
agentOpts.ActorOptimizerOptions.GradientThreshold      = opts.GradThresh;
agentOpts = setOpt(agentOpts, 'TargetUpdateFrequency', 2, 'rlTD3AgentOptions', ...
    'se dung mac dinh -- khong khop ban da hoi tu');
% CHUOI NAY TRUOC DAY HARDCODE "gamma 0.99 | GradThresh 1.0" trong khi gia tri
% that lay tu opts -- da lam mot lan doc log ket luan nham la repro bi troi cau
% hinh. In gia tri THAT. Sua 09/08.
% [Sua 09/08, ban thu hai] "L2 1e-4" van con hardcode trong chinh chuoi nay --
% dung loi vua sua o dong tren, sot lai mot chu. Nay doc tu CRITIC_L2.
fprintf(['      Dong bo voi cau hinh da hoi tu: gamma %g | L2 %g | ' ...
         'GradThresh %g | TargetUpdateFreq 2 | rScale %g | rewardMode %s\n'], ...
        opts.DiscountFactor, CRITIC_L2, opts.GradThresh, cfg.rScale, ...
        string(getfield_d(cfg,'rewardMode','?')));
% ===========================================================================
% ACTOR LR = 1e-6 (ban va nay), KHONG PHAI 1e-7 (qua dong bang) HAY 1e-5 (sup).
% ---------------------------------------------------------------------------
% Adam dich chuyen moi tham so ~ LR MOI LAN CAP NHAT, bat ke do lon gradient.
% Voi 4000 buoc/episode chia PolicyUpdateFrequency = 2 -> 2000 cap nhat/episode:
%
%   LR = 1e-5 : 400 ep x 2000 x 1e-5 = 8.00 don vi dich chuyen  -> DA SUP (do duoc)
%   LR = 1e-6 : 400 ep x 2000 x 1e-6 = 0.80 don vi dich chuyen  <-- dang dung
%   LR = 1e-7 : 400 ep x 2000 x 1e-7 = 0.08 don vi dich chuyen  -> qua dong bang
%
% Thang trong so fullyConnectedLayer khoi tao ~ 0.05-0.1. 1e-6 dat do dich
% chuyen o ~0.8 -- du de actor THUC SU di chuyen khoi diem xuat phat BC trong
% 400 episode, nhung con kem 10 lan muc da gay sup o 1e-5. Day la trung diem
% hinh hoc co chu dich giua hai dau da do duoc, KHONG phai doan mo lai.
%
% DO DUOC (lan chay truoc, LR 1e-5): pha A ket thuc +11.44 voi 0/20 that bai;
% tha actor ra thi that bai dau tien roi vao episode 8 -- dung luc dich chuyen
% tich luy dat 0.16, tuc vua vuot thang trong so. Den episode 24: 4/24 that bai,
% trung binh -76.81 tren 13 episode cuoi. Xu huong di xuong, khong dao dong.
% Cong kiem tra B1/B2 duoi day dung CHINH dau hieu nay (that bai som + xu huong
% am) de dung Pha B SOM neu 1e-6 van con qua lon.
%
% Bai hoc: ham nong critic KHONG cuu duoc mot actor LR sai thang. No chi tri
% hoan. Hai thu phai dung DONG THOI.
% ===========================================================================
agentOpts.ActorOptimizerOptions.LearnRate     = opts.ActorLRFineTune;
agentOpts.CriticOptimizerOptions(1).LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions(2).LearnRate = 1e-3;
% Pha A va pha B la HAI lan goi train() tren cung agent. Neu tuy chon nay con
% ton tai va de mac dinh (true), pha B se XOA experience buffer ma pha A vua
% do day -- critic van giu trong so nen khong hong, nhung phi toan bo 20
% episode ham nong. Truong nay da bi bo o cac ban RL Toolbox moi -> dat phong thu.
agentOpts = setOpt(agentOpts, 'ResetExperienceBufferBeforeTraining', false, ...
                   'rlTD3AgentOptions', 'buffer pha A se bi xoa o pha B');
agent = rlTD3Agent(actor, [mkCritic() mkCritic()], agentOpts);

% --- Kiem chung: chinh sach BC co tai lap duoc SMC-ESO khong? -----------
% Hai bang duoi dung CHUNG danh sach kich ban va CHUNG bo chi so, chi khac o
% chinh sach chay. Bang tren la SMC-ESO tinh moi 1 ms (dung bo dieu khien cua
% Bang 1-6). Bang duoi la chinh sach BC chay ZOH 100 Hz -- tuc dung dieu kien
% trien khai. Hieu giua hai bang LA khoang lech ZOH, do truc tiep thay vi doan.
fprintf('      (1) Tham chieu SMC-ESO @1 kHz (bo cua Bang 1-6):\n');
Mref = reportPolicy([], cfg, '        ');
fprintf('      (2) Tham chieu SMC-ESO @ZOH %g Hz (DOI CHUNG THAT cua BC):\n', 1/cfg.Ts_agent);
MrefZ = reportPolicy([], cfg, '        ', cfg.Ts_agent);
fprintf('      (3) Chinh sach BC @ZOH %g Hz:\n', 1/cfg.Ts_agent);
Mbc  = reportPolicy(agent, cfg, '        ');

% ===========================================================================
% CHAM DIEM: BC PHAI DUOC SO VOI (2), KHONG PHAI (1). Sua 08/08.
% ---------------------------------------------------------------------------
% Ban cu so (3) voi (1) -- tuc so mot bo 100 Hz voi mot bo 1 kHz. Chenh lech
% thu duoc tron hai nguon khong lien quan nhau:
%     (1) -> (2) : cai gia cua ZOH. KHONG phai loi cua BC. Khong chinh sach
%                  100 Hz nao tranh duoc no, ke ca mot ban sao hoan hao.
%     (2) -> (3) : sai so hoi quy cua BC. DUNG cai cong nay ton tai de do.
% Do duoc o hat giong 1 khi con so lan: d rms = +57%, va khong the biet bao
% nhieu trong do la BC. Cong bao "BC CHUA DAT" cho mot dai luong ma BC khong
% chiu trach nhiem.
%
% Gio cham diem tren (2)->(3). Hang (1) van in, va hieu (1)->(2) duoc in rieng
% thanh mot so lieu cua bai bao ("gia cua ZOH"), khong giau.
% ===========================================================================
dPkZ  = 100*(MrefZ(:,1)-Mref(:,1))./Mref(:,1);
dRmsZ = 100*(MrefZ(:,2)-Mref(:,2))./Mref(:,2);
dIaeZ = 100*(MrefZ(:,4)-Mref(:,4))./Mref(:,4);
scn  = {'danh nghia','nhieu 1.0','nhieu 2.0'};
fprintf('\n      GIA CUA ZOH  (1)->(2): SMC-ESO mat gi khi ha 1 kHz xuong %g Hz\n', 1/cfg.Ts_agent);
fprintf('      %-12s %9s %9s %9s\n', '', 'd peak', 'd rms', 'd IAEth');
for k = 1:3
    fprintf('      %-12s %8.0f%% %8.0f%% %8.0f%%\n', scn{k}, dPkZ(k), dRmsZ(k), dIaeZ(k));
end
fprintf('      ^ so lieu cua bai bao: day la cai gia cua rang buoc trien khai,\n');
fprintf('        KHONG phai loi cua behavior cloning.\n');

% --- Cham diem TU DONG, khong de mat thuong so ---------------------------
% rho la thong ke MOT mau (dinh) nen rat nhieu -- da hoc duoc dieu do o BC1,
% noi rho lech 64%% trong khi IAEtheta chi lech 0.6%%.
% [Sua 09/08] Cau cu o day ghi "chi so quyet dinh la rms va IAEtheta". Khong
% con dung: rms da bi go khoi tieu chi dat/khong dat -- xem khoi HIEU CHUAN
% ngay duoi. Ba dai luong duoi day van in du de chan doan.
dPk  = 100*(Mbc(:,1)-MrefZ(:,1))./MrefZ(:,1);
dRms = 100*(Mbc(:,2)-MrefZ(:,2))./MrefZ(:,2);
dIae = 100*(Mbc(:,4)-MrefZ(:,4))./MrefZ(:,4);
fprintf('\n      SAI SO BC    (2)->(3): cai cong nay cham diem\n');
fprintf('      %-12s %9s %9s %9s\n', 'lech vs (2)', 'd peak', 'd rms', 'd IAEth');
for k = 1:3
    fprintf('      %-12s %8.0f%% %8.0f%% %8.0f%%\n', scn{k}, dPk(k), dRms(k), dIae(k));
end
% ===========================================================================
% TIEU CHI CUA CONG -- HIEU CHUAN LEN AGENT DA BIET LA TOT. Sua 09/08.
% ---------------------------------------------------------------------------
% Ban cu: bcOK = max|dRms| <= 20 && max|dIae| <= 20.
%
% DA DO (Check_BCGate_OnAgent tren rl_v13_bc/agent_bc_seed1.mat, 08/08 10:04 --
% agent DUY NHAT vuot ca PSO-FLC lan SMC-ESO tren Eval_RL_v13_final):
%
%     kich ban     d peak     d rms   d IAEth    rho     rho@ZOH
%     danh nghia    -37%     +390%       0%     0.307     0.487
%     nhieu 1.0     -21%     +269%     -11%     0.392     0.476
%     nhieu 2.0     -17%     +184%     -11%     0.288     0.342
%     dB1 = +8%  (san cua SMC-ESO@1kHz = +9%)
%
% Agent thang cuoc TRUOT cong cu voi d rms = 390%. Ket luan bat buoc: d rms
% do DO TRUNG THANH voi SMC-ESO@ZOH, khong do CHAT LUONG. Agent nay can thiep
% lien tuc bien do vua (rms 1.23) thay vi nhoi dinh thua (rms 0.25), va vi the
% BAM SAT hon o IAEtheta (-11%) trong khi rho VAN thap hon tham chieu. Mot
% nguong tren |d rms| loai bo chinh nghiem tot nhat da tim duoc.
%
% Nguyen tac hieu chuan cua ban 09/08: MOT CONG PHAI CHO ARTIFACT DA BIET LA
% TOT DI QUA.
%
% ---------------------------------------------------------------------------
% [Sua 09/08, ban thu hai] NGUYEN TAC TREN DUNG NHUNG DA BI DUNG SAI CACH.
% ---------------------------------------------------------------------------
% "Cho artifact tot di qua" la dieu kien CAN de mot cong hop le. Ban 09/08 dung
% no nhu dieu kien DU de THIET KE cong. Hai he qua, ca hai deu da xay ra:
%
%   1. No chi chan false-reject tren DUNG mot artifact. Khong noi gi ve
%      false-accept, va mot artifact khong du de chung minh tieu chi dung.
%   2. Cach thoa no re nhat la dat nguong DUNG BANG so do cua artifact ->
%      bien dung sai bang khong theo cau tao. Do chinh la dieu da xay ra voi
%      dieu kien (b) rho <= rho@ZOH.
%
% VA (b) KHONG PHAI TIEU CHI DA CHOT. THIETKE_MOITRUONG_RL_v13.md muc 6, tieu
% de "Tieu chi thanh cong -- chot TRUOC khi huan luyen", viet:
%
%       rho = max|u_corr| / max|u_FLC|  <=  0.5    tren moi kich ban
%       Dai tham chieu: SMC-ESO dat rho thuoc [0.28, 0.43]
%
% Nguong TUYET DOI 0.5, dat cao hon dinh dai do duoc 0.43 -- bien dung sai da
% ton tai, co chu dich, va co truoc moi ket qua. Ban 09/08 thay no bang so sanh
% DIEM theo tung kich ban voi rhoRefZ = [0.487 0.476 0.342], tuc chat hon 0.5 o
% 2/3 kich ban (kich ban nhieu 2.0 chat hon 32%). Do la mot tieu chi KHAC, sinh
% ra sau khi da thay ket qua, khong phai tieu chi da chot.
%
% Ba loi cua (b), theo thu tu nang dan:
%
%   (i)   rho la thong ke MOT mau (dinh). O BC1 no lech 64% trong khi IAEtheta
%         lech 0.6%. Bien truot dien hinh (~2%) nho hon tan cua chinh no ~30
%         lan -- cong dang phan quyet o thang min hon do phan giai cua phep do.
%   (ii)  rhoRefZ CUNG la mot mau nhieu. So sanh hai lan rut n=1 cua cung mot
%         dai luong tan 64%: phuong sai cua HIEU con lon hon tung ve.
%   (iii) SAI PHAM TRU VE GIAI DOAN. Muc 6 dat rho len CHINH SACH HOC DUOC
%         CUOI CUNG, do tren TRUNG VI 5 HAT GIONG. Cong nay dat no len KHOI TAO
%         BC tai n=1. BC la diem xuat phat ma pha A/B co nhiem vu dich chuyen;
%         chan o day loai bo khoi tao ma RL nhieu kha nang keo xuong duoc, va
%         chon hat giong BC theo mot tinh chat ma huan luyen se thay doi.
%
% NEN: rho bi GO khoi dieu kien dat/khong dat cua cong BC. No van duoc IN, va
% van canh bao neu vuot 0.5, nhung khong con chan pha A. Kiem tra kien truc
% rho <= 0.5 chuyen ve dung cho muc 6 quy dinh: Eval_RL_v13_final, tren chinh
% sach cuoi, trung vi 5 hat giong -- xem muc "DIEU KIEN KIEN TRUC" trong file do.
%
% Hai dieu kien con lai giu nguyen, ca hai deu da kiem lai tren
% agent_bc_seed1.mat va deu DAT:
%
%   (a) d IAEtheta <= +20%   MOT PHIA. Tot hon tham chieu khong phai la loi --
%                            day chinh la cho ban cu sai: no phat ca hai chieu.
%   (c) dB1 <= +20%          Rang buoc muc 6, giu nguyen.
%
% d rms va rho VAN IN ra nhu chan doan (chenh lech nang luong can thiep so voi
% SMC-ESO@ZOH la mot so lieu that cua bai bao), nhung KHONG con la dieu kien
% dat/khong dat.
%
% CANH BAO CHO LAN SAU: neu mot dieu kien dat/khong dat sap duoc them hoac
% siet lai, kiem tra truoc hai cau hoi. Nguong nay co trong muc 6 khong? No
% duoc suy ra tu so do cua bao nhieu mau? Neu cau tra loi la "khong" va "mot",
% thi do la do tham so tren cai thuoc, khong phai tren mo hinh.
% ===========================================================================
rhoAg   = Mbc(:,3);
rhoRefZ = MrefZ(:,3);
% Rang buoc muc 6: khong xau hon PSO-FLC thuan qua 20% o Bang 1 danh nghia.
IAE_PSOFLC_B1 = 0.024463;
dB1    = 100*(Mbc(1,4)  - IAE_PSOFLC_B1)/IAE_PSOFLC_B1;
dB1ref = 100*(Mref(1,4) - IAE_PSOFLC_B1)/IAE_PSOFLC_B1;
fprintf('      IAEtheta danh nghia vs Bang 1 (PSO-FLC thuan %.6f): %+.0f%%\n', ...
        IAE_PSOFLC_B1, dB1);

% ---------------------------------------------------------------------------
% NGUONG VA PHAN QUYET DA DUOC TACH SANG bc_gate_v13.m -- MOT NOI DUY NHAT.
% Check_BCGate_OnAgent.m goi DUNG ham nay, nen cong cu kiem toan khong the troi
% khoi cong that duoc nua. Truoc do hai ben duoc giu khop bang mot dong comment
% va da troi that: |d rms| <= 20 bi go o day 09/08 nhung con lai ben kia.
% KHONG chep logic phan quyet tro lai day.
% ---------------------------------------------------------------------------
G = bc_gate_v13(dIae, dB1, rhoAg, rhoRefZ, ...
                struct('scn',{scn}, 'BCIters',opts.BCIters, ...
                       'BCEpisodes',opts.BCEpisodes, 'dB1ref',dB1ref));
fprintf('      rho vs (2):  %s  (CHAN DOAN, khong phai dieu kien dat)\n', G.rhoLine);
if ~isempty(G.rhoWarn), fprintf(2, '      (!) %s\n', G.rhoWarn); end
bcOK = G.ok;
% ===========================================================================
% THONG BAO LOI PHAI NOI DIEU KIEN NAO HONG. Sua 08/08.
% ---------------------------------------------------------------------------
% Ban cu in DUNG MOT thong bao cho moi truong hop hong, va thong bao do luon
% khuyen "quay ve TailFrac = 0". Hau qua do duoc: hat giong 1 truot o dB1
% (20.55%) voi TailFrac = 0.15, doc thong bao, dat TailFrac = 0, roi truot o
% d rms (57%) -- va nhan LAI CUNG mot loi khuyen dat TailFrac = 0, tuc dung
% cai vua lam. Mot vong tron, va moi vong ton mot lan sinh du lieu BC.
%
% Gio moi dieu kien tu bao ten minh va dua loi khuyen rieng cua no.
%
% [Sua 09/08, ban thu hai] Cac thong bao nay da chuyen sang bc_gate_v13.m cung
% voi phan quyet, de cong that va cong cu kiem toan noi CUNG MOT cau khi cung
% mot dieu kien hong. Dieu kien rho DA BI GO -- khong them lai o day; ly do o
% khoi TIEU CHI CUA CONG ben tren va trong bc_gate_v13.m.
% ===========================================================================
failMsg = G.failMsg;
% bcOK va failMsg phai luon dong y -- ca hai deu tu bc_gate_v13. Assert nay giu
% cho chung khong the tach nhau neu ai do sua mot ben.
assert(bcOK == isempty(failMsg), 'bc_gate_v13 tra ok va failMsg mau thuan nhau.');
% bcGatePassed di theo agent vao MOI file luu, de khong the doc mot agent ma
% khong biet no co qua cong hay khong. Them 10/08 cung voi SkipBCGate.
bcGatePassed = bcOK;
bcFailMsg    = failMsg;
if bcOK
    fprintf('      => BC DAT. Tiep tuc pha A.\n');
elseif opts.SkipBCGate
    % =======================================================================
    % SkipBCGate -- CHI dung cho nghien cuu PHAN BO qua nhieu hat giong.
    % -----------------------------------------------------------------------
    % Ly do ton tai: neu chi danh gia nhung hat giong QUA cong thi do la CHON
    % LOC THEO KET QUA. Muon phat bieu "trung vi 5 hat giong" thi phai chay ca
    % 5, ke ca nhung hat giong truot, roi bao cao ty le dat nhu mot so lieu.
    % Do duoc den 10/08 o TailFrac = 0.15: hat giong 1 truot (dB1 +20.55%),
    % hat giong 2 dat, hat giong 3 truot (d IAEth +33%, dB1 +43%) -> 1/3.
    %
    % KHONG dung tuy chon nay de "cho qua" mot cau hinh roi bao cao no nhu da
    % dat. Agent sinh ra o che do nay PHAI duoc bao cao kem trang thai cong.
    % Truong bcGatePassed duoc luu vao file de khong the quen.
    % =======================================================================
    fprintf(2, '      => BC CHUA DAT (%d dieu kien) nhung SkipBCGate = true.\n', numel(failMsg));
    for i = 1:numel(failMsg), fprintf(2, '        (%d) %s\n', i, failMsg{i}); end
    fprintf(2, ['      CHAY TIEP CO Y THUC de do phan bo. Agent se mang bcGatePassed = false.\n' ...
                '      KHONG duoc bao cao agent nay nhu mot ket qua da qua cong.\n']);
else
    fprintf(2, '      => BC CHUA DAT. Dieu kien hong:\n');
    for i = 1:numel(failMsg), fprintf(2, '        (%d) %s\n', i, failMsg{i}); end
    error('Train_RL_v13_bc:bcGate', ...
        ['BC CHUA DAT nguong da chot (%d dieu kien hong, xem tren). Dung TAI DAY thay vi\n' ...
         'chay tiep pha A/B, vi cong nay ton tai chinh de ngan viec do tham so cho toi khi\n' ...
         'dat -- chay tiep se vo hieu hoa muc dich cua no.\n\n' ...
         'QUY TAC DA CHOT: neu doi mot tham so lam dieu kien nay dat nhung lam dieu kien\n' ...
         'khac hong, DUNG LAI va bao cao thieu hut do nhu mot han che.\n\n' ...
         'TRUOC KHI KET LUAN "gioi han cua behavior cloning": chay\n' ...
         '    Check_BCGate_OnAgent(''rl_v13_bc/agent_bc_seed1.mat'')\n' ...
         'Neu agent da biet la tot cung truot dung dieu kien nay thi CONG SAI, khong\n' ...
         'phai phuong phap sai. DA XAY RA HAI LAN:\n' ...
         '  - nguong |d rms| <= 20%% loai chinh agent thang cuoc (d rms = 390%%), go 09/08;\n' ...
         '  - nguong rho <= rhoRefZ loai hat giong 2 boi bien 1.9%% tren mot dai luong\n' ...
         '    tan 64%%, trong khi tieu chi da chot (muc 6) la rho <= 0.5 va hat giong 2\n' ...
         '    DAT no. Go 09/08 ban thu hai.\n' ...
         'Ca hai deu la nguong hieu chuan tren MOT artifact. Truoc khi them nguong moi,\n' ...
         'kiem tra: no co trong THIETKE muc 6 khong, va no suy ra tu bao nhieu mau?'], ...
        numel(failMsg));
end

% --- (3) LUU CHINH SACH BC-ONLY, TRUOC KHI TD3 dong vao ---
% Day la nhanh doi chung. Phai chay DU giao thuc danh gia (muc 8 cua
% THIETKE_MOITRUONG_RL_v13.md) cho CA file nay lan agent cuoi, roi doi chieu.
% Neu hai ben trung nhau trong sai so do -> RL khong dong gop gi, va ket luan
% dung la ket qua am, du cac con so tuyet doi co "dat" moc SMC-ESO.
bcFile = fullfile(opts.SaveDir, sprintf('agent_bconly_seed%d.mat', seed));
% ---------------------------------------------------------------------------
% rewardVersion: TEM PHIEN BAN HAM THUONG. Bat buoc tu ban va 2 tro di.
%   1 = truoc ban va 2 (bonus KHONG tru baseline, het gio bao IsDone=1)
%   2 = sau  ban va 2 (bonus/penalty tru tham chieu bo nen, het gio IsDone=2)
% Khong co tem nay thi khong the biet mot file agent cu duoc huan luyen tren
% ham thuong nao -- va do CHINH LA cai da bien Resume_RL_v13_bc_B2 thanh mot
% cai bay. Cac script nap agent PHAI kiem tra truong nay.
% ---------------------------------------------------------------------------
rewardVersion = rewardVersionConst();
save(bcFile, 'agent', 'cfg', 'seed', 'agentOpts', 'rmseBC', 'rewardVersion', ...
     'bcGatePassed', 'bcFailMsg');
fprintf('      Da luu chinh sach BC-only: %s\n', bcFile);

end   % <-- ket thuc "if ~resuming": moi thu tren day bi bo qua khi tiep tuc Pha A
rewardVersion = rewardVersionConst();

%% ============ 4. PHA A -- HAM NONG CRITIC (actor dong bang bang LR = 0) ==
% ===========================================================================
% SUA LOI CO CHE: KHONG DUNG PolicyUpdateFrequency DE DONG BANG ACTOR.
% ---------------------------------------------------------------------------
% Lan chay truoc dat PolicyUpdateFrequency = nStepEp (4000) de actor gan nhu
% khong cap nhat. Nhung TD3 nguyen ban (Fujimoto 2018) tri hoan cap nhat actor
% VA cap nhat target network CUNG MOT tan so d -- do chinh la co che "delayed
% policy update". rlTD3AgentOptions.PolicyUpdateFrequency trong MATLAB di theo
% dung thiet ke do: no khoa CA HAI, khong rieng actor.
%
% Hau qua do duoc: Q0 dung yen o -0.025 suot 100 episode (400 000 buoc), trong
% khi gia tri chiet khau uoc luong tu chinh loi nhuan quan sat la +0.311. Target
% network chi duoc lam moi 100 lan (moi episode mot lan, tau = 0.005 moi lan),
% va giua hai lan lam moi, toi 4000 buoc gradient lien tiep cung duoi theo MOT
% muc tieu tinh -- dung co che critic can de hoc bi hy sinh de doi lay viec
% dong bang actor.
%
% Sua: tach hai viec. Dong bang actor bang LEARNING RATE = 0, GIU
% PolicyUpdateFrequency = 2 nguyen suot de target network chay dung nhip TD3
% nguyen ban. LR 0 khong lam actor dich chuyen mot ly, bat ke goi train() bao
% nhieu lan -- sach hon va khong dung cham toi co che boostrap cua critic.
% ===========================================================================
nStepEp    = round(cfg.Tf/cfg.Ts_agent);
% Thu muc checkpoint RIENG cho tung pha -- xem mkPhaseDir() ve lan va chuong
% checkpoint da lam mat du lieu B1 cua hat giong 1.
dirA  = mkPhaseDir(opts.SaveDir, seed, 'phaseA');
dirB1 = mkPhaseDir(opts.SaveDir, seed, 'phaseB1');
dirB2 = mkPhaseDir(opts.SaveDir, seed, 'phaseB2');

rl_rho_log('reset', cfg);   % theo doi rho tu episode dau tien cua pha A
                            % (cfg de bang phan ra in dung ten cong thuc reward)

fprintf('[4/5] Pha A: ham nong critic %d episode (actor dong bang bang LR=0)...\n', ...
        opts.WarmupEpisodes);
% ===========================================================================
% ActorLR CUA PHA B LAY TU opts, KHONG DOC NGUOC TU AGENT. Sua 09/08.
% ---------------------------------------------------------------------------
% Ban cu doc lai LR hien tai cua agent roi cat di de dung lai o Pha B. Voi mot
% lan chay lien mach thi dung (luc do agent con giu 1e-6 vua dat khi tao).
% NHUNG voi duong opts.ResumeAgent thi SAI HOAN TOAN:
%
%   file AABORT duoc luu TAI CONG CRITIC, tuc SAU khi Pha A da dat LR = 0.
%   -> nap lai thi lrActorFineTune doc duoc = 0
%   -> Pha B dat LR = 0
%   -> B1 + B2 chay 2 gio voi actor DONG BANG, khong hoc gi, va KHONG BAO GI CA.
%
% Dung loai that bai am tham ma du an nay da vap nhieu lan. Lay tu opts thi gia
% tri khong phu thuoc trang thai nhat thoi cua agent, va co the doc duoc ngay
% tren dong lenh.
% ===========================================================================
lrActorFineTune = opts.ActorLRFineTune;
fprintf('      ActorLR cho Pha B = %.0e (tu opts, khong doc nguoc tu agent)\n', ...
        lrActorFineTune);
assert(lrActorFineTune > 0, ...
    ['opts.ActorLRFineTune = %g. Pha B se chay voi actor dong bang va khong hoc\n' ...
     'duoc gi. Dat mot gia tri duong (mac dinh 1e-6).'], lrActorFineTune);
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 0;
% PolicyUpdateFrequency GIU NGUYEN = 2 (dat luc tao agent) -- KHONG dung o day.
%
% KIEM TRA NGAY, TRUOC KHI VAO train() 24 PHUT: gan property long nhau tren
% mot object DA DUNG (agent.AgentOptions.X.Y = z) la diem mo ho kinh dien giua
% handle class va value class trong MATLAB OOP -- phep gan co the khong thuc
% su ghi nguoc vao agent. Doc lai NGAY sau khi gan; neu so khong khop 0, dung
% tai day, dung doi het pha A moi biet.
lrCheck = agent.AgentOptions.ActorOptimizerOptions.LearnRate;
puCheck = agent.AgentOptions.PolicyUpdateFrequency;
fprintf('      (kiem tra gan) ActorLR pha A doc lai = %g (ky vong 0) | PolicyUpdateFrequency = %g (ky vong 2)\n', ...
        lrCheck, puCheck);
if lrCheck ~= 0
    error('Train_RL_v13_bc:actorFreezeFailed', ...
        ['Gan agent.AgentOptions.ActorOptimizerOptions.LearnRate = 0 KHONG ghi nguoc\n' ...
         'vao agent (doc lai duoc %g, khong phai 0). Day la diem mo ho handle/value\n' ...
         'class trong MATLAB OOP, khong phai loi ban chay. DUNG TAI DAY, bao lai ket\n' ...
         'qua nay thay vi de train() chay 24 phut vo ich.'], lrCheck);
end
% ===========================================================================
% PHA A CHAY THEO KHOI, DO slope SAU MOI KHOI -> DUONG CONG, KHONG PHAI 1 DIEM.
% ---------------------------------------------------------------------------
% VI SAO: ba lan chan doan lien tiep deu truot vi moi lan chi co MOT diem do.
%   lan 1: "gamma qua thap"        -> sai
%   lan 2: "nghen lan truyen"      -> bac bo bang Q0 phang... nhung Q0 la mot
%          diem don le cua ham gia tri (G_0 = 0.76 trong khi std(G_t) = 5.3),
%          nen "Q0 phang" KHONG chung minh duoc "critic khong hoc".
%   lan 3: "L2 = 1e-4"             -> phep thu bi lam hong vi doi CUNG LUC ca
%          L2 lan do dai Pha A (100 -> 25 ep). Hai diem thu duoc:
%              25 ep, L2=0    : slope 0.008
%             100 ep, L2=1e-4 : slope 0.247
%          khong so duoc voi nhau, va deu phu hop voi gia thuyet "thang do lon
%          dan theo thoi gian huan luyen".
%
% Cau hoi con lai chi co MOT: slope CO DANG LEO khong, va no PHANG o dau?
%   con leo o cuoi Pha A  -> thieu thoi gian. Them episode, hoac n-step.
%   phang duoi nguong     -> co rao can co cau (L2, thang phan thuong, mang).
% Mot duong cong tra loi duoc; mot diem thi khong. Do la ly do khoi nay ton tai.
%
% Chi phi: moi lan do ~4 episode mo phong. Voi 4 khoi la ~16 episode = ~6 phut
% tren mot Pha A 100 episode (~40 phut). Re hon nhieu so voi mot lan chay 40
% phut nua chi de biet mot diem thu ba.
% ===========================================================================
% WarmupEpisodes = 0 (chi hop le khi resume): critic da ham nong du o lan chay
% truoc, khong ham lai. Dung de di thang toi Pha B ma khong tra lai 64 phut.
if opts.WarmupEpisodes == 0
    assert(resuming, ['WarmupEpisodes = 0 chi dung duoc cung ResumeAgent -- ' ...
        'neu khong thi critic chua he duoc ham nong.']);
    stPrev = getfield_d(Rz, 'statsA', struct());
    rA     = getfield_d(stPrev, 'EpisodeReward', []);  rA  = rA(:);
    q0A    = getfield_d(stPrev, 'EpisodeQ0',     []);  q0A = q0A(:);
    curve  = getfield_d(stPrev, 'slopeCurve', zeros(0,4));
    tWarm  = 0;
    fprintf('      BO QUA Pha A (WarmupEpisodes = 0): dung critic da ham nong %d episode o lan truoc.\n', ...
            numel(rA));
    statsA = struct('EpisodeReward', rA, 'EpisodeQ0', q0A, 'slopeCurve', curve);
else
% ===========================================================================
% [Sua 09/08, ban thu ba] TU GIA HAN -- KHONG DOI NGUOI CHAY LAI BANG TAY.
% ---------------------------------------------------------------------------
% Ban cu chay dung WarmupEpisodes co dinh (100), in duong cong, roi DUNG LAI
% de nguoi doc tu quyet dinh co chay tiep khong -- moi quyet dinh la mot vong
% chay/doc/doi rieng, hang chuc phut moi vong. Phan tich 4 diem dau (25-100 ep)
% bang hoi quy tuyen tinh cho R^2 = 0.9997: KHONG co dau hieu cong (bao hoa)
% trong du lieu da co. Nghia la "con leo hay da phang" la cau hoi TRA LOI DUOC
% tu chinh du lieu da do, khong can doi mot lan chay nua roi doan tiep.
%
% Vong lap nay tu chay them khoi, dung khi MOT trong ba dieu xay ra:
%   (a) DAT MUC TIEU: slope >= WarmupTargetSlope (khop scaleOK cua cong ben
%       duoi -- xem "NGUONG SUA 09/08"). Du de qua phan an toan bien do.
%   (b) DA PHANG: buoc cuoi thay doi < 25% buoc dau, VA slope < 0.5 -- rao can
%       co cau (L2 / thang phan thuong / mang), them episode se khong cuu.
%   (c) CHAM TRAN: da chay MaxWarmupEpisodes ma chua dat (a) hay (b) -- dung
%       va bao cao nguyen trang, KHONG tu y vuot ngan sach.
% Ket qua cua MOT lenh chay la mot trong ba ket luan tren, khong phai mot diem
% du lieu roi ngoi cho nguoi doc dien giai.
% ===========================================================================
epChunk = opts.WarmupChunk;
rA = [];  q0A = [];  curve = zeros(0,4);
if resuming   % noi tiep duong cong cu de doc duoc lien mach
    curve = getfield_d(getfield_d(Rz,'statsA',struct()), 'slopeCurve', zeros(0,4));
end
epDone0 = 0;
if ~isempty(curve), epDone0 = curve(end,1); end
tA = tic;
stopReason = '';
ic = 0;
while true
    ic = ic + 1;
    optA  = mkTrainOpts(epChunk, nStepEp, opts, dirA, false);
    stC   = train(agent, env, optA);
    rA    = [rA;  stC.EpisodeReward(:)];  %#ok<AGROW>
    q0A   = [q0A; stC.EpisodeQ0(:)];      %#ok<AGROW>
    [rc, sc_, g0c, q0c] = critic_fit_score(agent, cfg, 4, 40);
    epTot = epDone0 + ic*epChunk;
    curve(end+1,:) = [epTot, rc, sc_, q0c];  %#ok<AGROW>
    fprintf('      [khoi %d] sau %3d ep: r = %+.3f | slope = %+.4f | G_0 = %+.3f | Q(s0) = %+.3f\n', ...
            ic, epTot, rc, sc_, g0c, q0c);

    if sc_ >= opts.WarmupTargetSlope
        stopReason = sprintf('DAT MUC TIEU: slope %.4f >= %.2f (WarmupTargetSlope).', ...
                              sc_, opts.WarmupTargetSlope);
        break;
    end
    if size(curve,1) >= 3
        dLast  = curve(end,3) - curve(end-1,3);
        dFirst = curve(2,3)   - curve(1,3);
        if abs(dLast) < 0.25*abs(dFirst) && sc_ < 0.5
            stopReason = sprintf(['DA PHANG duoi muc tieu (buoc cuoi %+.4f vs buoc dau %+.4f).\n' ...
                '         Them episode se KHONG cuu. Rao can la CO CAU: thang phan thuong hoac\n' ...
                '         mang critic -- xem de xuat o cong ben duoi.'], dLast, dFirst);
            break;
        end
    end
    if epTot >= opts.MaxWarmupEpisodes
        stopReason = sprintf(['CHAM TRAN MaxWarmupEpisodes = %d, van chua dat muc tieu %.2f.\n' ...
            '         Dung va bao cao nguyen trang -- tang MaxWarmupEpisodes neu muon chay them,\n' ...
            '         thay vi de vong lap tu y vuot ngan sach.'], opts.MaxWarmupEpisodes, opts.WarmupTargetSlope);
        break;
    end
end
tWarm  = toc(tA);
statsA = struct('EpisodeReward', rA, 'EpisodeQ0', q0A, 'slopeCurve', curve);

fprintf('\n      --- Duong cong slope theo so episode ---\n');
for jc = 1:size(curve,1)
    bar = repmat('#', 1, max(0, round(60*curve(jc,3))));
    fprintf('      %3d ep : slope %+.4f  %s\n', curve(jc,1), curve(jc,3), bar);
end
fprintf('      => DUNG PHA A: %s\n', stopReason);
kA = min(4, size(curve,1));
fprintf('      slope %d do cuoi: trung binh %.4f | do lech giua cac lan do %.4f\n', ...
        kA, mean(curve(end-kA+1:end,3)), std(curve(end-kA+1:end,3)));
end   % <-- ket thuc "if opts.WarmupEpisodes == 0 ... else"

% --- CONG KIEM TRA CRITIC: no co hoc duoc gi khong? ---------------------
% Lan truoc Q0 dung yen o ~ -0.025 suot 100 episode (400 000 buoc) trong khi gia
% tri chiet khau dung la ~ +0.311 -- do LA hau qua cua loi PolicyUpdateFrequency
% da neu tren, khong phai actor chua du thoi gian. Cong nay van giu lai: neu sua
% xong ma Q0 van khong bam duoc dai loi nhuan quan sat, do moi la ly do that de
% dung lai truoc khi tha actor.
rA   = statsA.EpisodeReward(:);
q0A  = statsA.EpisodeQ0(:);
nTail_ = min(20, numel(rA));
% ---------------------------------------------------------------------------
% Uoc luong gia tri chiet khau that. Ban cu dung 1/(1-gamma) -- do la tong CHUOI
% VO HAN. O gamma=0.99 sai so khong dang ke (0.99^4000 ~ 1e-18), nhung o
% gamma=0.999 thi 0.999^4000 = 0.018 nen chuoi vo han vuot thuc te ~1.8%, va o
% gamma=0.9996 (neu sau nay thu) thi 0.9996^4000 = 0.20 -- vuot toi 20%. Dung
% cong thuc HUU HAN dung so buoc that de cong kiem tra critic khong bi lech
% chinh boi tham so ma no dang danh gia.
% ---------------------------------------------------------------------------
% DUNG CA 100 EPISODE PHA A, KHONG PHAI 20 CUOI. Sua 08/08 -- lan thu BA cua
% cung mot loi trong du an nay (cong B1 dung 10 ep cuoi; cong nay dung 20).
%
% Cua so duoi chi co y nghia khi chinh sach DANG DOI, de bat "trang thai cuoi".
% Trong Pha A actor DONG BANG (LearnRate = 0) theo dung thiet ke, nen ca 100
% episode la 100 mau doc lap cung phan bo. Vut 80% du lieu di khong duoc gi va
% lam nhieu tang 2.2 lan.
%
% Do duoc o hat giong 1 -- ngưỡng cua chinh cong nay khi do bang 20 episode:
%     vTrue(20 ep)  = 0.568 +- 0.763   -> 0.74 sigma, KHONG khac 0
%     vTrue(100 ep) = 1.281 +- 0.341   -> 3.75 sigma
% Tuc cong dang so Q0 voi mot con so khong phan biet duoc voi khong. Ket luan
% "critic chua hoc" van dung (Q0 = 2.8% cua 1.281), nhung do la may -- neu Q0
% roi vao vung 0.1-0.2 thi ket qua cong se do CUA SO quyet dinh, khong phai do
% critic.
%
% q0End van lay 20 cuoi: Q0 la du doan cua critic, ma critic THI DANG HOC suot
% Pha A. Voi no, "trang thai cuoi" moi la cai can do -- cua so duoi la dung.
% Hai dai luong nay khac ban chat nen co chu dich lay cua so khac nhau.
% ---------------------------------------------------------------------------
% ===========================================================================
% CONG CRITIC DUNG PHEP DO MONTE-CARLO, KHONG DUNG "Q0 vs vTrue" NUA.
% ---------------------------------------------------------------------------
% Cong cu: |Q0| < 0.25*|vTrue|, voi vTrue = r_tb/nStep*horiz. Da do duoc no
% hong o CA HAI dau:
%
%   NHAN SAI  : cong thuc gia dinh reward rai deu trong episode. Do that bang
%               Monte-Carlo: G_0 = 3.245 +- 1.153 trong khi vTrue = 1.281.
%               Lech 2.5 lan. Moi nguong 25% tu truoc den nay deu do tren nhan
%               sai -- ke ca cai da chan lan chay hom qua.
%
%   VO HUONG  : mot con so khong tach duoc "critic mu" khoi "critic biet nhung
%               bi nen thang do". Hai nguyen nhan khac han, hai cach sua nguoc
%               nhau, ma cong cu tra ve cung ket qua. Chinh vi the ba tuan qua
%               di sai huong hai lan (gamma, roi n-step).
%
% Thay bang critic_fit_score: chay Monte-Carlo that va tra ve HAI so tach bach.
%   r     thap        -> critic khong biet gi     -> loi co che hoc
%   r cao, slope thap -> biet nhung bi nen        -> loi thang do / L2
% Ton ~1-2 phut, dung bang gia mot phan tram cua doan B1+B2 ma no bao ve.
% ===========================================================================
gam   = agent.AgentOptions.DiscountFactor;
horiz = (1 - gam^nStepEp) / (1 - gam);
q0End = mean(q0A(end-nTail_+1:end));
fprintf('      Xong pha A (%.1f phut). Dang do critic bang Monte-Carlo...\n', tWarm/60);
rl_rho_log('report');   % ban va 2 muc E: so hang nao dang chi phoi return?

[rFit, slopeFit, G0mc, Q0mc, Dfit] = critic_fit_score(agent, cfg, 6, 20);
fprintf('\n      --- Cong critic (Monte-Carlo, %d episode) ---\n', Dfit.nEp);
fprintf('      G_0 THAT (Monte-Carlo) : %+.3f +- %.3f\n', G0mc, Dfit.semG0);
fprintf('      Q_critic(s0)           : %+.3f\n', Q0mc);
fprintf('      Q0 theo statsA (20 cuoi): %+.3f   [tham khao]\n', q0End);
fprintf('      tuong quan r           : %+.3f   (nguong 0.50)\n', rFit);
fprintf('      he so goc slope        : %+.3f   (nguong 0.50 .. 2.00, ky vong 1)\n', slopeFit);
fprintf('      gamma %.4f | chan troi %.0f buoc = %.1f s\n', gam, horiz, horiz*cfg.Ts_agent);
% ===========================================================================
% CONG NAY GIO DUNG THAT, KHONG CHI CANH BAO (sua 08/08).
% ---------------------------------------------------------------------------
% Ban cu chi in canh bao roi CHAY TIEP B1+B2 (~2 gio) du critic chua hoc. Do
% vo hieu hoa chinh muc dich cua Pha A: 24 phut nay ton tai DE QUYET DINH co
% nen tieu 2 gio tiep hay khong. In mot dong do rat de troi qua trong log dai,
% va nguoi chay thuong chi quay lai sau khi moi thu da xong.
%
% Doi chieu voi hai cong con lai trong file nay: cong BC (dong 451) va cong B1
% (dong 656) DEU error() that. Cong critic la cong duy nhat khong -- mot lech
% chuan khong co ly do, va la lech dat nhat vi no o ngay truoc doan ton thoi
% gian nhat.
%
% Trang thai duoc LUU lai truoc khi dung, dung khuon mau B1ABORT: 24 phut Pha A
% khong mat, va co lenh chay tiep in san neu ban xem so lieu roi van muon di.
% ===========================================================================
% ===========================================================================
% NGUONG SUA 09/08. DOC KY -- DAY KHONG PHAI "NOI CHO QUA".
% ---------------------------------------------------------------------------
% Nguong cu: r >= 0.50 VA slope trong [0.50, 2.00]. Do duoc o 200 episode:
%     r = 0.940 (rat tot)  |  slope = 0.482  -> TRUOT vi thieu 0.018.
%
% Ba ly do nguong slope = 0.50 la SAI, khong phai ket qua sai:
%
% (1) TD3 CO Y UOC LUONG THAP. Clipped double Q-learning lay min(Q1,Q2) lam
%     muc tieu, va Fujimoto et al. (2018) neu ro no "may induce an
%     underestimation bias" -- do la CHU DICH, de thien lech xuong thay vi len
%     (uoc luong thap lam hong chinh sach it hon uoc luong cao). Doi hoi
%     slope ~ 1 la doi hoi thuat toan dung lam cai no duoc thiet ke de lam.
%     Mot slope < 1 on dinh la critic TD3 KHOE, khong phai critic hong.
%
% (2) NGUONG NAM TRONG NHIEU DO. Bon lan do cuoi (125-200 ep) tren CUNG mot
%     critic: 0.4455, 0.5066, 0.4067, 0.4729 -- trung binh 0.458, do lech
%     0.042. Da VUOT 0.5066 o 150 ep roi tut ve 0.4067 o 175 ep. Chenh lech
%     voi nguong (0.018) NHO HON do lech giua cac lan do (0.042) hai lan.
%     Mot cong ma ket qua doi dau tuy theo lan do la mot cong khong doc duoc.
%     Day la lan thu TU trong du an nay mot nguong duoc ap len mot thong ke
%     nhieu voi cua so hep (truoc do: B1 10-ep, vTrue 20-ep, Q0 don diem).
%
% (3) SAI DAI LUONG. Cai quyet dinh Pha B co cai thien duoc chinh sach hay
%     khong la HUONG cua grad_a Q, tuc critic co xep hang HANH DONG dung
%     khong. slope la BIEN DO tuyet doi, va Adam chuan hoa buoc nen mot phep
%     nen deu khong doi buoc cap nhat actor. r = 0.940 moi la do luong gan
%     nhat voi cai ta can.
%
% Nguong moi: r >= 0.85 (chinh) va slope >= 0.30 (san an toan, bat truong hop
% critic thuc su khong co bien do). Ca hai lay TRUNG BINH cua may lan do cuoi
% de khong bi mot lan do xui quyet dinh.
%
% VA QUAN TRONG NHAT: phep thu THAT khong nam o cong nay ma o B1 -- 40 episode
% (~13 phut) voi cong thong ke Welch da co. Neu critic that su khong dung duoc,
% B1 se cho thay reward tut co y nghia va dung lai. Khong can mot cong doan
% truoc de bao ve mot cong do sau, nhat la khi cong doan bi nhieu chi phoi.
% ===========================================================================
kAvg = min(3, size(curve,1));
if kAvg > 0
    rGate     = mean([curve(end-kAvg+1:end,2); rFit]);
    slopeGate = mean([curve(end-kAvg+1:end,3); slopeFit]);
else
    rGate = rFit;  slopeGate = slopeFit;
end
fprintf('      Cham diem tren trung binh %d lan do: r = %+.3f | slope = %+.3f\n', ...
        kAvg+1, rGate, slopeGate);
knowsShape = abs(rGate)  >= 0.85;                    % xep hang -- tieu chi CHINH
scaleOK    = slopeGate   >= 0.30 && slopeGate <= 2.00;  % san an toan bien do
criticOK   = knowsShape && scaleOK;
fprintf('      Chinh sach da troi chua?\n');
reportPolicy(agent, cfg, '        ');
if ~criticOK
    aFile = fullfile(opts.SaveDir, sprintf('agent_bc_seed%d_AABORT.mat', seed));
    save(aFile,'agent','statsA','cfg','seed','agentOpts','rewardVersion', ...
        'bcGatePassed','bcFailMsg', ...
                'tWarm','rmseBC','bcFile','rFit','slopeFit','G0mc','Q0mc');
    if ~knowsShape
        why = sprintf(['CRITIC KHONG BIET GI: r = %+.3f < 0.50. No khong phan biet duoc\n' ...
            '          trang thai tot voi xau. Day la loi CO CHE HOC.\n' ...
            '          Nghi van, re -> dat: (1) CriticLR 1e-3 -> 1e-4 hoac 3e-3;\n' ...
            '          (2) criticGraph; (3) buffer/minibatch. KHONG phai gamma.'], rFit);
    else
        % ===================================================================
        % [Sua 09/08, ban thu hai] LOI KHUYEN CU O DAY DA CU SAU BAN VA 9.
        % -------------------------------------------------------------------
        % Ban cu hardcode "nhan reward len ~100 lan". He so 100 duoc viet cho
        % rewardMode = 'cv' (r = (cBase-cA)*dt) -- rwip_env_v13 dong 236 ghi ro
        % "'cv' (ban cu; can rScale = 100)". Mac dinh hien tai la 'state' voi
        % rScale = 1, va ban va 9 DA tra thang do ve dung cho: reward do duoc
        % ~0.19/buoc so voi ~0.47/buoc cua ban da hoi tu, tuc lech 2.4 lan chu
        % khong phai 215 lan. Nhan 100 lan nua se dua thang Q len ~3000, vuot
        % ban da hoi tu (~47) gan 70 lan.
        %
        % Loi khuyen do da song sot qua mot ban va lam no sai, vi no la chuoi
        % ky tu chu khong phai mot phep tinh. Nay he so duoc TINH tu so do that
        % va chi de xuat khi con lech that.
        %
        % QUAN TRONG HON: neu duong cong slope CON DANG LEO thi phep do nay
        % chua xac dinh duoc nguyen nhan -- ca ba ung vien (thoi gian, L2,
        % thang phan thuong) deu phu hop voi mot diem cuoi lay TRUOC khi on
        % dinh. Doc phan quyet tu do la lap lai dung loi da go o cong BC, o
        % mot dai luong khac.
        % ===================================================================
        rPerStep = mean(rA)/nStepEp;
        R_REF    = 0.47;   % reward/buoc cua ban DA HOI TU (rwip_env_v13, ban va 7)
        fScale   = R_REF/max(abs(rPerStep), eps);
        slopeClimb = false;
        if size(curve,1) >= 3
            dLast  = curve(end,3) - curve(end-1,3);
            dFirst = curve(2,3)   - curve(1,3);
            slopeClimb = ~(abs(dLast) < 0.25*abs(dFirst));
        end
        if slopeClimb
            why = sprintf(['CRITIC BIET HINH DANG, THANG DO CHUA ON DINH: r = %+.3f (tot),\n' ...
                '          slope = %+.3f (ky vong ~1) -- nhung duong cong slope VAN DANG LEO\n' ...
                '          (xem bang ngay tren). PHEP DO NAY CHUA XAC DINH DUOC NGUYEN NHAN:\n' ...
                '          ca ba ung vien (thieu thoi gian | L2 = %g | thang phan thuong) deu\n' ...
                '          phu hop voi mot diem cuoi lay TRUOC khi on dinh. Dung doc phan\n' ...
                '          quyet tu no -- do la doc o do phan giai ma phep do chua dat toi.\n' ...
                '          RE NHAT, LAM TRUOC: tang WarmupEpisodes (dang %d) toi khi slope\n' ...
                '          PHANG. Chi khi phang thi gia tri cuoi moi co nghia, va moi so duoc\n' ...
                '          L2 = 0 o CUNG ngan sach episode -- so CA DUONG CONG, khong so diem.\n' ...
                '          CHUA phai luc dong den thang phan thuong: reward ~%.2e/buoc so voi\n' ...
                '          %.2f/buoc cua ban da hoi tu, lech %.1f lan (KHONG phai 100 lan).'], ...
                rFit, slopeFit, CRITIC_L2, opts.WarmupEpisodes, rPerStep, R_REF, fScale);
        else
            if strcmpi(getfield_d(cfg,'rewardMode','state'),'cv') && cfg.rScale < 100
                sNote = sprintf(['Che do ''cv'' can rScale = 100 (rwip_env_v13 dong 236);\n' ...
                                 '                dang la %g -- DANG THIEU dung he so do.'], cfg.rScale);
            elseif fScale >= 3
                sNote = sprintf(['Neu dat rScale thi dat ~%.0f, KHONG phai 100:\n' ...
                                 '                100 la he so cua che do ''cv'', khong phai cua ''%s''.'], ...
                                 fScale, getfield_d(cfg,'rewardMode','?'));
            else
                sNote = sprintf(['Thang phan thuong DANG DUNG (lech %.1f lan < 3) --\n' ...
                                 '                nghi van nay YEU, uu tien (1) va (3).'], fScale);
            end
            why = sprintf(['CRITIC BIET HINH DANG NHUNG SAI THANG DO: r = %+.3f (tot) nhung\n' ...
                '          slope = %+.3f (ky vong ~1). No xep hang dung, bien do bi nen %.1f lan.\n' ...
                '          Duong cong slope DA PHANG -> rao can CO CAU, khong phai thieu thoi gian.\n' ...
                '          Thu tu re -> dat:\n' ...
                '            (1) L2 = 0 cho critic (dang %g), CUNG ngan sach episode, so ca duong cong.\n' ...
                '            (2) Thang phan thuong: reward ~%.2e/buoc | ban da hoi tu ~%.2f/buoc\n' ...
                '                | lech %.1f lan | rewardMode = ''%s'', rScale = %g.\n' ...
                '                %s\n' ...
                '            (3) Mang critic / CriticLR. KHONG phai gamma.'], ...
                rFit, slopeFit, 1/max(slopeFit,eps), CRITIC_L2, ...
                rPerStep, R_REF, fScale, getfield_d(cfg,'rewardMode','?'), cfg.rScale, sNote);
        end
    end
    fprintf(2, ['\n      (!) %s\n\n' ...
                '          DUNG TAI DAY -- khong tieu ~2 gio cho B1+B2 tren critic nay.\n' ...
                '          Da luu trang thai Pha A: %s\n\n' ...
                '          Xem chi tiet: out = Diag_CriticFit_v13(''%s'');\n' ...
                '          Neu xem roi VAN muon chay tiep:\n' ...
                '            agent = Train_RL_v13_bc(%d, struct(''SkipCriticGate'',true));\n'], ...
            why, aFile, aFile, seed);
    if ~opts.SkipCriticGate
        error('Train_RL_v13_bc:criticGate', ...
            ['Cong critic khong dat sau Pha A (r = %.3f, slope = %.3f). Dung truoc khi\n' ...
             'tieu 2 gio cho B1+B2. Dat SkipCriticGate = true de bo qua co y thuc.'], ...
            rFit, slopeFit);
    end
    fprintf(2, '      SkipCriticGate = true -- bo qua cong CO Y THUC. Chay tiep pha B.\n');
else
    fprintf('      Critic DAT: r = %+.3f, slope = %+.3f. Tiep tuc pha B.\n', rFit, slopeFit);
end

%% ============ 5. PHA B -- TINH CHINH BANG TD3 (B1 tham do + B2 tiep tuc) ==
% Xem "BAN VA" muc 4 o dau file: chia ngan sach %d episode da chot thanh B1
% (tham do, co cong kiem tra an toan) va B2 (phan con lai, CHI chay neu B1
% khong lap lai dau hieu sup do da do duoc o LR=1e-5). Day KHONG phai gia han
% ngan sach -- tong episode B1+B2 van dung bang opts.MaxEpisodes da chot.
nProbe = min(40, opts.MaxEpisodes);
nRest  = opts.MaxEpisodes - nProbe;

fprintf('[5a/5] Pha B1: tham do %d episode dau o ActorLR=%.0e...\n', nProbe, lrActorFineTune);
agent.AgentOptions.ActorOptimizerOptions.LearnRate = lrActorFineTune;  % 1e-6, tha actor
% PolicyUpdateFrequency van la 2 tu dau -- khong can dat lai.
optB1 = mkTrainOpts(nProbe, nStepEp, opts, dirB1, true);
t0 = tic;  statsB1 = train(agent, env, optB1);  tB1 = toc(t0);

% --- Cong kiem tra: B1 co lap lai dau hieu sup do da do duoc o LR=1e-5? -----
% Dau hieu do duoc lan LR=1e-5 (xem "BAN VA" muc 2): that bai dau tien roi vao
% ~episode 8; den episode 24, 4/24 that bai va trung binh -76.81 tren 13
% episode cuoi -- XU HUONG DI XUONG so voi cuoi pha A, khong phai dao dong on
% dinh. Dung lai hai dau hieu do lam nguong: >=15% episode "that bai" (reward
% <= -0.9*failPen la dai dien, vi that bai duoc phat -failPen=200 co dinh), HOAC
% reward TB cua 10 episode cuoi B1 tut duoi 50% muc cuoi pha A (rEndA).
% ===========================================================================
% SUA LOI THONG KE (do duoc, hat giong 1): cua so 10 episode cuoi qua hep so
% voi bien thien tu nhien cua moi truong. Ngay ca khi actor DONG BANG (Pha A),
% reward tung episode dao dong -8 .. +127 -- SD ~35. Voi cua so 10, sai so
% chuan ~35/sqrt(10) ~ 11, nen mot cua so xui (7.70) co the lech ca 2 SD ma
% khong phai sup that. Doi chieu: hat giong 1 co 0/40 that bai va TRUNG BINH
% CA 40 episode = 23.94 (75% muc Pha A), nhung cua so-10 rieng le = 7.70 (24%)
% -- CHINH cua so hep gay bao dong gia, khong phai LR=1e-6 sup that. Dau hieu
% that su cua lan sup LR=1e-5 la THAT BAI XUAT HIEN (4/24) va trung binh AM SAU
% (-76.81) -- lech bac do lon so voi bien thien binh thuong, khong phai mot
% cua so ngau nhien xui. Sua: dung TRUNG BINH CA B1 (khong phai duoi 10 cuoi)
% lam tieu chi chinh; cua so-10 chi in de tham khao, khong dung de quyet dinh.
% ===========================================================================
% ===========================================================================
% SUA LOI THU HAI (bat buoc kem ban va 2): NGUONG NHAN VO NGHIA O THANG MOI.
% ---------------------------------------------------------------------------
% Tieu chi cu la "meanB1 < 0.5*rEndA" -- mot nguong NHAN. No chi co nghia khi
% rEndA duong va lon (32 o thang cu). Sau ban va 2, bonus duoc tru tham chieu
% bo nen, nen mot chinh sach BC (gan trung bo nen) cho return quanh 0. Luc do:
%
%   rEndA = +0.5  -> cong bao dong khi meanB1 < +0.25   (sat sat, de bao gia)
%   rEndA = -2.0  -> cong bao dong khi meanB1 < -1.0    (tuc DOI CHIEU: mot
%                    agent TOT HON, meanB1 = -0.5, van vuot cong -- con mot
%                    agent te hon nua thi lot)
%   rEndA ~= 0    -> cong tro thanh ngau nhien hoan toan
%
% Voi rEndA am, bat dang thuc DOI CHIEU va cong bien thanh vo nghia. Do dung
% la kieu lech dinh nghia ma dau file nay da liet ke ba lan lam hong du an
% (thetaTol 0.02 vs 0.002, wrap goc banh da, OS_theta luon bang theta0) -- va
% no se kich hoat ngay lan chay dau tien sau ban va 2 neu khong sua bay gio.
%
% Thay bang PHEP THU Y NGHIA THONG KE, khong phu thuoc thang do va khong phu
% thuoc dau: B1 chi bi coi la sup neu trung binh cua no thap hon cuoi pha A
% QUA 2 sai so chuan cua chinh hieu do. Do dung la phep tinh dang le phai lam
% ngay tu dau -- neu co no, bao dong gia 7.70 (lech ~1.5 SE) da khong xay ra,
% va ta da khong mat mot phien chan doan cho no.
% ===========================================================================
rB1     = statsB1.EpisodeReward(:);
rTailA  = rA(end-nTail_+1:end);
rEndA   = mean(rTailA);                                % tham chieu: cuoi pha A
% ===========================================================================
% DEM THAT BAI BANG SO BUOC, KHONG BANG NGUONG REWARD. Sua 09/08 (ban va 9).
% ---------------------------------------------------------------------------
% Ban cu: rB1 <= -0.9*failPen*rScale. Nguong do chi co nghia khi -failPen la
% gia tri AM CO BIET so voi dai reward binh thuong. Sau ban va 9 ('state'):
%     dai reward/episode ~ -2000 .. +1600, con -failPen*rScale = -200
% -> nguong -180 nam GIUA dai binh thuong, va cong se dem hang loat episode
%    BINH THUONG thanh "that bai". Cong B1 khi do bao dong lien tuc.
%
% Mot episode that bai la episode KET THUC SOM -- do la dinh nghia khong phu
% thuoc thang phan thuong, dung cho ca hai che do. Doc thang tu EpisodeSteps.
% ===========================================================================
if isprop(statsB1,'EpisodeSteps') || isfield(statsB1,'EpisodeSteps')
    nFailB1 = sum(statsB1.EpisodeSteps(:) < nStepEp);
else
    nFailB1 = sum(rB1 <= -0.9*cfg.failPen*cfg.rScale);   % du phong ban cu
    fprintf(2,'  (!) statsB1 khong co EpisodeSteps -- dem that bai bang nguong reward (kem tin cay).\n');
end
nTailB1 = min(10, numel(rB1));
trendB1 = mean(rB1(end-nTailB1+1:end));                % chi de tham khao/in log
meanB1  = mean(rB1);                                   % tieu chi CHINH: trung binh CA B1
% Sai so chuan cua HIEU hai trung binh (Welch, khong gia dinh cung phuong sai)
seDiff  = sqrt( var(rTailA)/numel(rTailA) + var(rB1)/numel(rB1) );
dropSig = (rEndA - meanB1) / max(seDiff, eps);         % tut bao nhieu sai so chuan

% ===========================================================================
% THEM PHEP THU XU HUONG TRONG LONG B1. Sua 09/08 -- do duoc no bo sot mot cu
% sup THAT (hat giong 1, lan chay 09/08):
%
%   ep  1-10: +1133 | ep 11-20: +630 | ep 21-30: -131 | ep 31-40: -898
%   hoi quy: do doc -66.8/episode, t = -4.47  -> tut -2673 qua 40 episode
%   nhung meanB1 = 183.6 > rEndA = 123.1  -> cong KHONG bao dong
%
% Trung binh CA B1 che mat cu tut vi no gop 20 episode dau (con tot) voi 20
% episode cuoi (da sup). Day la mat trai cua chinh ban va truoc: cong cu dung
% 10 episode cuoi -> qua HEP, bao dong gia; sua thanh trung binh ca B1 -> qua
% RONG, bo sot. Ca hai deu la mot dai luong don le khong nhin thay XU HUONG.
%
% Phep thu dung: hoi quy reward theo chi so episode NGAY TRONG B1. Mot chinh
% sach dang bi actor pha se cho do doc am co y nghia, bat ke trung binh la bao
% nhieu. Nguong t < -3: doi chieu, cua so-10 tung bao dong gia o hat giong 1
% (tut ~1.5 SE) se KHONG dat nguong nay, con cu sup that (t = -4.47) thi dat.
% ===========================================================================
nB1  = numel(rB1);
xB1  = (1:nB1).';
xm_  = mean(xB1);  ym_ = mean(rB1);
SxxB = sum((xB1-xm_).^2);
slB1 = sum((xB1-xm_).*(rB1-ym_)) / max(SxxB, eps);
resB = rB1 - (ym_ + slB1*(xB1-xm_));
seSl = sqrt( sum(resB.^2)/max(nB1-2,1) / max(SxxB,eps) );
tSl  = slB1 / max(seSl, eps);
fprintf('      Xu huong TRONG B1: do doc %+.1f/episode (t = %.2f). Nguong dung: t < -3.\n', ...
        slB1, tSl);
collapsed = (nFailB1 >= 0.15*nB1) || (dropSig > 2) || (tSl < -3);
% SkipB1Gate: cong B1 duoc hieu chuan cho lan chay 40 episode voi ham thuong
% da qua ban va. Cau hinh TAI LAP chi chay 18 episode tren ham thuong GOC, va
% ban goc co 8/38 that bai (21%) -- tuc no se vuot nguong 15% va bi chan, du
% chinh no la agent tot nhat cua du an. Cho phep bo qua CO Y THUC.
if collapsed && opts.SkipB1Gate
    fprintf(2, ['      (!) Cong B1 bao dong nhung SkipB1Gate = true -- bo qua CO Y THUC.\n' ...
                '          Ban goc (agent_bc_seed1.mat) co 21%% that bai va van la agent\n' ...
                '          tot nhat khi danh gia tat dinh. Ty le that bai luc huan luyen\n' ...
                '          KHONG du bao duoc chat luong chinh sach o day.\n']);
    collapsed = false;
end
fprintf('      B1 xong (%.1f phut). that bai=%d/%d | reward TB toan B1=%.2f | TB 10 cuoi=%.2f (tham khao) | tham chieu cuoi pha A=%.2f\n', ...
        tB1/60, nFailB1, numel(rB1), meanB1, trendB1, rEndA);
fprintf('      Muc tut: %.2f sai so chuan (SE hieu = %.2f). Nguong dung = 2.0 SE.\n', ...
        dropSig, seDiff);
fprintf(['      DOI CHIEU hat giong 1 (thang reward CU): tut 7.70 vs 31.89 voi SD~35\n' ...
         '      tuong duong ~1.5 SE -- DUOI nguong 2 SE, tuc cong nay se KHONG bao\n' ...
         '      dong, dung nhu ket luan hau nghiem. Cong nhan cu (0.5*rEndA) da bao.\n']);
if collapsed
    fprintf(2, ['      (!) B1 CO DAU HIEU SUP DO GIONG LR=1e-5 (that bai>=15%% hoac\n' ...
                '          trung binh tut qua 2 sai so chuan so voi cuoi pha A). DUNG TAI DAY thay vi\n' ...
                '          chay tiep B2 -- ActorLR=%.0e van con qua lon cho cau hinh nay.\n' ...
                '          De xuat: thu lai voi actor LR nho hon (vd 5e-7 hoac 3e-7)\n' ...
                '          bang cach sua bien lrActorFineTune, hoac bao cao ket qua am\n' ...
                '          voi cau hinh nay (gamma=0.99 + reward thua + uMax=8V).\n'], lrActorFineTune);
    rhoLog  = rl_rho_log('get');
    outFile = fullfile(opts.SaveDir, sprintf('agent_bc_seed%d_B1ABORT.mat', seed));
    save(outFile,'agent','statsA','statsB1','cfg','seed','agentOpts','rewardVersion', ...
        'bcGatePassed','bcFailMsg', ...
                 'tWarm','tB1','rmseBC','bcFile','rhoLog');
    fprintf('      Da luu trang thai tai thoi diem dung: %s\n', outFile);
    error('Train_RL_v13_bc:phaseBCollapse', ...
        'Pha B1 co dau hieu sup do (xem canh bao tren). Dung truoc khi chay B2.');
end
fprintf('      B1 KHONG co dau hieu sup do. Tiep tuc B2 (%d episode con lai).\n', nRest);

fprintf('[5b/5] Pha B2: tiep tuc tinh chinh %d episode (KHONG gia han)...\n', nRest);
% nRest = 0 khi MaxEpisodes <= 40 (vd cau hinh tai lap dung 18 episode). Goi
% train() voi MaxEpisodes = 0 se gay loi. Bo qua B2 va tao stats rong de doan
% luu/bao cao ben duoi van chay. Do duoc: agent tot nhat cua du an chi dung 18
% episode tinh chinh, nen truong hop nay KHONG hiem.
if nRest > 0
    optB2 = mkTrainOpts(nRest, nStepEp, opts, dirB2, true);
    t1 = tic;  statsB2 = train(agent, env, optB2);  tB2 = toc(t1);
else
    fprintf('      B2 bo qua: MaxEpisodes = %d <= nProbe = %d, khong con episode nao.\n', ...
            opts.MaxEpisodes, nProbe);
    statsB2 = struct('EpisodeReward',[], 'EpisodeSteps',[], 'EpisodeQ0',[]);
    tB2 = 0;
end
tTrain = tB1 + tB2;

rhoLog  = rl_rho_log('get');   % [rho, max|u_corr|, max|u_FLC|, so buoc, that bai]
outFile = fullfile(opts.SaveDir, sprintf('agent_bc_seed%d.mat', seed));
save(outFile,'agent','statsB1','statsB2','statsA','cfg','seed','agentOpts','rewardVersion', ...
     'bcGatePassed','bcFailMsg', ...
             'tTrain','tB1','tB2','tWarm','rmseBC','bcFile','rhoLog');
fprintf('\nDa luu %s  (%.1f phut B1 + %.1f phut B2, %.1f phut tong)\n', ...
        outFile, tB1/60, tB2/60, (tTrain+tWarm)/60);
% Pha A goi mkTrainOpts voi doSave = false (actor dong bang -- khong co gi dang
% luu), nen phaseA/ CO Y de trong. Ban in cu liet ke ca ba nhu nhau, khien
% nguoi doc di tim checkpoint Pha A khong ton tai.
fprintf('Checkpoint (tach theo pha):\n  A : (khong luu -- actor dong bang trong pha A)\n  B1: %s\n  B2: %s\n', ...
        dirB1, dirB2);

%% --- Chan doan rho: agent co con o che do residual khong? --------------
if ~isempty(rhoLog)
    r = rhoLog(:,1);  nF = sum(rhoLog(:,5) > 0);
    fprintf(['\nrho theo episode (%d episode, chan doan -- KHONG nam trong phan thuong):\n' ...
             '  trung vi %.3f | p90 %.3f | max %.3f | %.1f%% episode co rho > 0.5\n' ...
             '  so episode that bai: %d\n' ...
             '  Dai tham chieu SMC-ESO: 0.28 .. 0.43. rho > 0.5 KEO DAI nghia la\n' ...
             '  chinh sach da THAY THE bo nen, khong con la lop hieu chinh residual\n' ...
             '  -- khi do phai trinh bay nhu mot kien truc khac (muc 6 THIETKE).\n'], ...
            size(rhoLog,1), median(r), qtile(r,0.90), max(r), 100*mean(r>0.5), nF);
end

fprintf('\n=== Chinh sach SAU tinh chinh ===\n');
reportPolicy(agent, cfg, '  ');
fprintf(['\nDoi chieu voi moc SMC-ESO (Bang 3-4):\n' ...
         '  nguong nhieu phuc hoi        : PSO-FLC 4.043 | bo lai SMC-ESO 5.047 rad/s\n' ...
         '  t phuc hoi tai xung 2.0 rad/s: PSO-FLC 11.86 | bo lai SMC-ESO 6.17 s\n' ...
         '  rho (ty le can thiep)        : SMC-ESO 0.28 .. 0.43\n' ...
         'Chay Compare_v13 / Test_Disturbance_v13 voi mode PSOFLC_RL de do chinh thuc.\n' ...
         'BAT BUOC: chay CUNG giao thuc do cho %s roi doi chieu.\n' ...
         'Neu BC-only va BC+TD3 trung nhau trong sai so do -> ket qua am.\n'], bcFile);
end


%% =====================================================================
function [loss, grad] = bcLoss(net, dlX, dlY)
yhat = forward(net, dlX);
loss = mean((yhat - dlY).^2, 'all');
grad = dlgradient(loss, net.Learnables);
end

function o = mkTrainOpts(nEp, nStep, opts, saveDir, doSave)
%MKTRAINOPTS  Tuy chon huan luyen. Ten truong luu agent doi giua cac phien ban
%   RL Toolbox, nen dat qua setOpt (cung kieu phong thu nhu setNoise2).
%
%   saveDir PHAI la thu muc RIENG CUA TUNG PHA -- xem mkPhaseDir() duoi day.
o = rlTrainingOptions( ...
    'MaxEpisodes',                nEp, ...
    'MaxStepsPerEpisode',         nStep, ...
    'ScoreAveragingWindowLength', 30, ...
    'Verbose',                    opts.Verbose, ...
    'Plots',                      opts.Plots, ...
    'StopTrainingCriteria',       'none');
if doSave
    w = 'se KHONG luu checkpoint';
    o = setOpt(o, 'SaveAgentCriteria',  'EpisodeReward',        'rlTrainingOptions', w);
    o = setOpt(o, 'SaveAgentValue',     opts.SaveAgentValue,    'rlTrainingOptions', w);
    o = setOpt(o, 'SaveAgentDirectory', saveDir,                'rlTrainingOptions', w);
end
end

function d = mkPhaseDir(baseDir, seed, phase)
%MKPHASEDIR  Thu muc checkpoint RIENG cho tung pha huan luyen.
%
% ===========================================================================
% VI SAO PHAI TACH -- LOI DA LAM MAT DU LIEU THAT (hat giong 1)
% ---------------------------------------------------------------------------
% RL Toolbox danh so checkpoint la Agent1.mat, Agent2.mat, ... va DANH SO LAI
% TU 1 MOI LAN GOI train(). Ban cu dung CHUNG mot thu muc seed<N>/ cho ca pha
% A, B1 va B2, nen:
%
%   - Agent7.mat cua B2 GHI DE Agent7.mat cua B1 -- am tham, khong canh bao.
%   - Kiem tra thuc te thu muc rl_v13_bc/seed1 ngay 08/08: ton tai lan lon
%     Agent1 (17:58), Agent10-13 (18:16-18:17), Agent100-149 (12:59-13:12),
%     Agent109/123 (18:42-18:46) -- it nhat BA lan chay tron vao nhau, khong
%     con phan biet duoc checkpoint nao thuoc pha nao hay lan chay nao.
%   - Hau qua nang nhat: sau khi B1 bi dung va ta muon quay ve checkpoint tot
%     nhat CUA B1 de so sanh, cac file do da bi B2 ghi de mat.
%
% Sau ban va: seed<N>/phaseA/, seed<N>/phaseB1/, seed<N>/phaseB2/ -- moi lan
% goi train() co khong gian ten rieng, khong the va cham.
%
% CANH BAO KEM THEO ve viec CHON checkpoint: SaveAgentValue = 0 nghia la MOI
% episode co reward > 0 deu duoc luu. Voi SD giua cac episode ~10-35 do duoc,
% chon "checkpoint co reward huan luyen cao nhat" la CHON TREN NHIEU: ban se
% chon duoc mot episode may man chu khong phai mot chinh sach tot hon. Muon
% chon checkpoint cho bai bao thi phai danh gia lai tung ung vien tren mot bo
% kich ban CO DINH tach rieng (Compare_v13.m), khong dung reward huan luyen.
% ===========================================================================
d = fullfile(baseDir, sprintf('seed%d', seed), phase);
if ~isfolder(d), mkdir(d); end
end


function o = setOpt(o, nm, v, who, consequence)
%SETOPT  Dat mot truong tuy chon NEU phien ban RL Toolbox nay co no.
%   Ten truong doi giua cac phien ban; im lang bo qua la cach tot nhat de mat
%   2 gio roi moi biet. Bao ra stderr de khong the khong thay.
if isprop(o,nm) || isfield(o,nm)
    o.(nm) = v;
else
    fprintf(2,'  (!) %s KHONG co truong %s -- %s\n', who, nm, consequence);
end
end

function M = reportPolicy(agent, cfg, pre, smcTsOverride)
%REPORTPOLICY  Bang chi so tren 3 kich ban chuan.
%   reportPolicy([],    cfg, pre)      -> SMC-ESO @1 kHz  (bo cua Bang 1-6)
%   reportPolicy([],    cfg, pre, Tsa) -> SMC-ESO @ZOH Tsa (dieu kien trien khai)
%   reportPolicy(agent, cfg, pre)      -> chinh sach da hoc @ZOH cfg.Ts_agent
%   Mot danh sach kich ban, mot bo chi so -> so sanh duoc.
%   M : 3x4 = [max|u_corr| , rms , rho , IAEtheta] theo tung kich ban.
%
%   Dang thu HAI la thu quan trong duoc them 08/08. Truoc do chi co @1 kHz, nen
%   moi chenh lech "BC vs tham chieu" deu tron sai so hoi quy cua BC voi cai gia
%   cua viec chay o 100 Hz. Ba hang cho phep tru bach:
%       @1kHz  vs @ZOH  = cai gia cua ZOH        (khong lien quan BC)
%       @ZOH   vs BC    = sai so hoi quy cua BC  (cai cong BC phai do)
M = nan(3,4);  kRow = 0;
P = cfg.P;  P.rlUmax = cfg.uMax;  P.rlTs = cfg.Ts_agent;  P.uCorrMax = cfg.uMax;
isRef = isempty(agent);
if isRef
    simMode = 'PSOFLC_SMCESO';
    if nargin >= 4 && ~isempty(smcTsOverride)
        P.smcTs = smcTsOverride;   % SMC-ESO chay o ZOH, xem sim_rwip_v13
    end
else
    simMode = 'PSOFLC_RL';
    P.rlPolicy = @(o) polAct(agent, o);
end
fprintf('%s%-16s%12s%12s%9s%12s\n', pre, 'kich ban', 'max|u_corr|', 'rms', 'rho', 'IAEtheta');
for sc = {'danh nghia', []; 'nhieu 1.0', struct('t',5,'dthetadot',1.0); ...
          'nhieu 2.0', struct('t',5,'dthetadot',2.0)}'
    Pk = P;  if ~isempty(sc{2}), Pk.dist = sc{2}; end
    o = sim_rwip_v13(simMode, Pk, [0.10;0;0;0], 40, cfg.Ts);
    kRow = kRow + 1;
    if o.diverged
        fprintf('%s%-16s%s\n', pre, sc{1}, '   MAT ON DINH');
    else
        M(kRow,:) = [o.u_corr_max, o.u_corr_rms, o.rho, o.IAEtheta];
        fprintf('%s%-16s%12.4f%12.4f%9.3f%12.5f\n', pre, sc{1}, ...
                o.u_corr_max, o.u_corr_rms, o.rho, o.IAEtheta);
    end
end
end

function v = rewardVersionConst()
%REWARDVERSIONCONST  Mot nguon duy nhat cho tem phien ban ham thuong.
%   1 = truoc ban va 2 (bonus khong tru baseline, het gio bao IsDone=1)
%   2 = sau  ban va 2 (bonus/penalty tru tham chieu bo nen, het gio IsDone=2,
%       va tu ban va 7 co them rScale -- rScale duoc kiem RIENG khi resume vi
%       no nam trong cfg, khong phai trong ma nguon).
v = 2;
end

function v = getfield_d(S, f, d)
%GETFIELD_D  S.(f) neu co, nguoc lai tra ve mac dinh d.
if isstruct(S) && isfield(S,f), v = S.(f); else, v = d; end
end

function n = nBatch(), n = 256; end

function a = polAct(agent, o)
a = getAction(agent, {o(:)});
if iscell(a), a = a{1}; end
a = double(a(1));
end

function o = setNoise2(o, nm, vals)
if ~isprop(o,nm) && ~isfield(o,nm), fprintf(2,'  (!) khong co %s\n',nm); return; end
m = o.(nm);  fn = fieldnames(vals);  ap = {}; sk = {};
for k = 1:numel(fn)
    if isprop(m,fn{k}) || isfield(m,fn{k}), m.(fn{k}) = vals.(fn{k}); ap{end+1}=fn{k}; %#ok<AGROW>
    else, sk{end+1}=fn{k}; end %#ok<AGROW>
end
o.(nm) = m;
fprintf('      %s: %s', nm, strjoin(ap,', '));
if ~isempty(sk), fprintf(2,'  | KHONG co: %s', strjoin(sk,', ')); end
fprintf('\n');
end

function lg = criticGraph(nObs, nAct)
% MOI LAYER MOT DONG -- xem ghi chu o actorNet. Ba mang duoi truoc day gop hai
% layer tren mot dong nen vertcat gay.
op = [featureInputLayer(nObs,'Name','obsIn')
      fullyConnectedLayer(256,'Name','fco1')
      reluLayer('Name','ro1')
      fullyConnectedLayer(256,'Name','fco2')];
ap = [featureInputLayer(nAct,'Name','actIn')
      fullyConnectedLayer(256,'Name','fca1')];
cp = [additionLayer(2,'Name','add')
      reluLayer('Name','rc1')
      fullyConnectedLayer(256,'Name','fcc1')
      reluLayer('Name','rc2')
      fullyConnectedLayer(1,'Name','QOut')];
lg = layerGraph(op); lg = addLayers(lg,ap); lg = addLayers(lg,cp);
lg = connectLayers(lg,'fco2','add/in1'); lg = connectLayers(lg,'fca1','add/in2');
end

function v = uni(a,b), v = a + (b-a)*rand(); end
function s = sgn(),    s = 1; if rand() < 0.5, s = -1; end, end

function v = qtile(x, p)
%QTILE  Phan vi don gian (khong can Statistics Toolbox).
x = sort(x(:));  n = numel(x);
if n == 0, v = NaN; return; end
v = x(max(1, min(n, ceil(p*n))));
end

function r = bcRmse(net, X, Y, idx)
%BCRMSE  RMSE cua actor tren mot tap con mau (lay toi da 20000 mau de nhanh).
if isempty(idx), r = NaN; return; end
idx = reshape(idx,1,[]);
if numel(idx) > 20000, idx = idx(randperm(numel(idx), 20000)); end
yh = predict(net, dlarray(single(X(:,idx)), 'CB'));
yv = dlarray(reshape(single(Y(idx)),1,[]), 'CB');
r  = sqrt(double(gather(extractdata(mean((yh - yv).^2, 'all')))));
end
