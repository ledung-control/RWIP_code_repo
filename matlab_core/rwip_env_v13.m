function [env, cfg] = rwip_env_v13(cfg)
%RWIP_ENV_V13  Moi truong huan luyen lop hieu chinh RL cho RWIP.
%
%   [env, cfg] = rwip_env_v13()        -- dung cau hinh mac dinh
%   [env, cfg] = rwip_env_v13(cfg)     -- ghi de mot so truong
%
% ===========================================================================
% TOM TAT THIET KE  (chi tiet va ly do: THIETKE_MOITRUONG_RL_v13.md)
% ---------------------------------------------------------------------------
% Kien truc:  V = sat( u_FLC + u_corr , +-24 V ),  u_corr = 8*a,  a in [-1,1]
%             Vong ngoai cascade va bo nen PSO-FLC v13 GIU NGUYEN.
%
% Quan sat  : 6 chieu, xem rl_obs_v13.m (mot nguon duy nhat, dung chung voi
%             sim_rwip_v13 khi danh gia).
%
% Phan thuong: r = ( c_base - c_agent ) * dt  +  bonus - penalty
%             c = |th|/0.010 + 0.054*|phi|/0.100
%                 + 0.05*(u_corr/uMax)^2 + 0.02*(du_corr/uMax)^2
%             c_base lay tu quy dao BO NEN chay song song tren CUNG dieu kien
%             dau, CUNG chuoi nhieu, CUNG tham so. Vi c_base khong phu thuoc
%             hanh dong cua agent, tru no di la phep tru baseline xac dinh:
%             GIAM PHUONG SAI ma KHONG DOI chinh sach toi uu (control variate).
%             Can thiet vi bo nen v13 da rat tot -> chi phi tho gan nhu khong
%             doi du agent lam gi, gradient hoc rat yeu.
%
% Dang L1 cho theta va phi (khong phai binh phuong): kiem tra K2 cho thay dang
% binh phuong chuan hoa theo dung sai xac lap khien so hang phi ap dao so hang
% theta 876 lan o nhieu 2.0 rad/s, va ty le bien thien 51 lan giua cac kich
% ban -> khong ton tai trong so co dinh nao can bang duoc. L1 giam bien thien
% xuong 5.1 lan. w_phi = 0.054 la trung binh nhan tren 6 kich ban.
%
% Episode : 40 s, agent 100 Hz (4000 buoc), doi tuong tich phan 1 kHz.
%           Cu hich trong [3,10] s: do duoc banh da can ~25 s de hap thu va
%           tra lai xung 2.0 rad/s, nen kich muon hon se bi cat episode giua
%           luc dang phuc hoi.
% ===========================================================================
%
% ===========================================================================
% BAN VA -- SUA THEO CHAN DOAN TD3 KHONG HOI TU (phien lam viec voi nguoi dung,
% xem doi thoai ve "TD3 that bai" va so sanh voi bai CEAI cung tac gia)
% ---------------------------------------------------------------------------
% Hai thay doi trong file nay, DOC LAP voi thay doi DiscountFactor/ActorLR o
% Train_RL_v13_bc.m (ca hai file phai sua CUNG LUC, xem ghi chu ben do):
%
% (A) uMax: 10 V (42% bien 24 V) -> 8 V (~33% bien). Ngan sach cu qua rong so
%     voi quy uoc "residual" trong tai lieu (vd Johannink 2019), va gan bang
%     ca ngan sach SMC-ESO -- kho phan biet "hieu chinh nho" voi "thay the".
%     SMC-ESO trong make_P_v13.m GIU NGUYEN uCorrMax=10 (bo do da duoc chinh
%     dinh RIENG cho ngan sach do, doi so se lam sai lech ket qua Bang 3-4 da
%     cong bo) -- chi ngan sach cua LOP RL thay doi.
%
%     SUA LAI TU 5V (do duoc, lan chay dau tien cua ban va nay): tai uMax=5V,
%     BC KHONG DAT cong o Buoc 2 -- KHONG PHAI vi TailFrac sai, ma vi SMC-ESO
%     (dung lam nhan bat chuoc) tu no da can toi 4.1647V/5V (83%) NGAY CA O
%     KICH BAN DANH NGHIA (khong nhieu), va 95.8%/100% o hai kich ban nhieu.
%     Tuc 5V la muc tran CHU DONG bao hoa SMC-ESO gan nhu lien tuc, khong con
%     la mot "residual" ma la mot rang buoc dang xieng ca bo dieu khien giai
%     tich. Bao hoa lam nhan bat chuoc tro nen bam-bang/cat cut o hau het quy
%     dao -- mang 256-256 ReLU hoi quy tron muot mot tin hieu nhu vay, gay lech
%     lon o vong kin (do duoc: d rms +155%/+96%/+56%, dung du du dau hieu "1 su
%     kien bao hoa" cua muc 8 THIETKE, khong phai loi hieu chuan TailFrac). 8V
%     giu con so bao hoa danh nghia o ~52%% (4.1647/8) -- con du dia, van nho
%     hon dang ke muc 10V/42%% ban dau.
%
% (B) Phan thuong: them SO HANG THUA (bonus/penalty nguong), ben canh phan
%     dense (c_base - c_agent)*dt van giu nguyen. Bai CEAI cung tac gia (kien
%     truc 4-ngo-vao, TD3 hoi tu) dung ca hai loai. Ly do them: gan diem hoi
%     tu, chenh lech dense giua hai chinh sach gan-toi-uu tien ve 0, tin hieu
%     thua bien do lon moi con phan biet duoc "tot" voi "rat tot". Tham so
%     r_bonus=+10 (giu bam sat 1 giay lien tuc) / r_penalty=-5 (vua vuot nguong
%     phan ky, canh bao SOM hon nguong that bai 0.5 rad) lay THEO DUNG quy uoc
%     bien do cua CEAI de so sanh duoc nguyen nhan that bai giua hai thiet ke.
% ===========================================================================
%
% ===========================================================================
% BAN VA 2 -- CHAN DOAN "CRITIC KHONG HOC" (Q0 ~ 0 vs gia tri that ~ +0.80)
% ---------------------------------------------------------------------------
% Doc tu rhoLog cua agent_bc_seed1_B1ABORT.mat (140 episode Pha A + B1):
%
%   TAT CA 140/140 episode chay du 4000 buoc. KHONG episode nao that bai.
%   KHONG episode nao ket thuc som. rho on dinh 0.315 -> 0.336 (actor khong sup).
%
% Hai he qua truc tiep tu quan sat do:
%
% (C) LOI TRUNCATION -- 140/140 EPISODE.
%     Dong cuoi stepFcn truoc day: isdone = ls.k >= numel(ls.cBase), tra ve
%     LOGICAL TRUE khi het gio. RL Toolbox phan biet:
%         IsDone = 1 -> TERMINAL, target = r          (KHONG bootstrap)
%         IsDone = 2 -> TRUNCATED, target = r + gamma*Q'  (CO bootstrap)
%     Het gio o t=40s KHONG phai trang thai terminal -- con lac van dang dung.
%     Khai bao nham lam critic hoc rang gia tri tai t=40s dung bang MOT buoc
%     reward cuoi (~0.008). Vi khong episode nao that bai, 100% episode dinh
%     loi nay.
%
%     O gamma=0.99 o nhiem lui theo 0.99^k nen chi ban ~100 buoc cuoi (2.5%).
%     O gamma=0.999 no voi nguoc 1000 buoc = 25% episode. Day la LY DO dong nay
%     PHAI duoc sua TRUOC khi nang gamma, khong phai sau.
%
% (D) SO HANG THUA NUOT TIN HIEU, VA NO KHONG DUOC TRU BASELINE.
%     Vi khong episode nao that bai va khong episode nao ket thuc som, -failPen
%     CHUA TUNG xuat hien trong 560 000 buoc, va moi bien thien reward chi co
%     the den tu bonus. reward/episode = 10*N_bonus + dense, dense la so O(1).
%     Quan sat -8 .. +127 => N_bonus = 0..13, trung binh 31.89 => ~3.2.
%     BONUS CHIEM ~90-95% RETURN.
%
%     Do la mau thuan truc tiep voi thiet ke: phan dense (c_base - c_agent) la
%     control variate, co CHU DICH triet phuong sai (xem "Phan thuong" o tren --
%     "bo nen v13 da rat tot -> gradient hoc rat yeu"). Ban va (B) them mot so
%     hang lon hon 1250 lan MOI BUOC va KHONG tru baseline, tuc bom nguoc lai
%     dung cai phuong sai ma dense vua kho cong khu. Te hon: inTight (|th|<=0.010
%     VA |ph|<=0.050 lien tuc 1 s) bi chi phoi boi KICH BAN BOC THAM (kick
%     0.5-3.0 rad/s, thoi diem 3-10 s) chu khong boi hieu chinh cua agent. SD~35
%     giua cac episode CHINH LA xo so nay, khong phai tin hieu hoc duoc.
%
%     SUA: giu NGUYEN y dinh cua ban va (B) -- tin hieu thua bien do lon van can
%     khi dense tien ve 0 gan diem hoi tu -- nhung ap DUNG PHEP TRU BASELINE cho
%     no, giong hach phan dense:
%
%         reward = (cB - cA)*dt  +  (bonus - bonusBase)  -  (penalty - penBase)
%
%     bonusBase/penBase lay tu CHINH quy dao bo nen da chay song song trong
%     baseRollout (cung x0, cung nhieu, cung tham so). Vi chung khong phu thuoc
%     hanh dong cua agent, phep tru KHONG DOI chinh sach toi uu -- dung lap luan
%     control-variate da dung cho dense. Nhung no xoa sach phan "bo nen tinh co
%     roi vao hop tight vi kich ban de", tuc phan xo so. Cai con lai dung bang
%     "agent bam sat DUOC BAO NHIEU LAU HON bo nen" -- do moi la thu agent dieu
%     khien duoc.
%
%     Kiem chung mien phi: voi action == 0 (agent trung het bo nen), CA BA so
%     hang phai trieu tieu ve ~0. Xem Test_Env_RewardDecomp_v13.m.
%
% (E) Ghi log phan ra reward (dense/bonus/penalty rieng) qua rl_rho_log, de lan
%     sau khong phai suy luan bang phep loai tru nua.
% ===========================================================================

if nargin < 1, cfg = struct(); end

%% ---------------- Cau hinh mac dinh ----------------
d = struct();
d.Tf        = 40;          % s, do dai episode
d.Ts        = 1e-3;        % s, buoc tich phan doi tuong
d.Ts_agent  = 1e-2;        % s, chu ky goi agent (ZOH)
d.uMax      = 8;           % V, bien hanh dong RL (residual that su -- xem BAN VA o dau file)
d.thetaFail = 0.5;         % rad, nguong coi nhu mat kha nang phuc hoi
d.failPen   = 200;         % phat khi that bai

% ===========================================================================
% BAN VA 7 -- rScale: NHAN TOAN BO PHAN THUONG. Bang chung tu MOT CAU HINH DA
% HOI TU tren cung lop doi tuong (WRIP_TD3_Training.m cua chinh tac gia).
% ---------------------------------------------------------------------------
% Doi chieu thang phan thuong giua hai cau hinh:
%
%                          DA HOI TU          FILE NAY (truoc ban va)
%   do dai episode         1500 buoc (15 s)   4000 buoc (40 s)
%   SaveAgentValue         700                0
%   => return/episode      > 700              ~8.8
%   => reward/buoc         ~0.47              ~0.0022      <-- nho hon ~215 lan
%   gamma                  0.99               0.999
%   => thang Q             ~47                ~2
%   L2RegularizationFactor 1e-4 (CO, van hoi tu)  1e-4
%
% Ban cau hinh da hoi tu GIU L2 = 1e-4 va van chay tot. Nghia la L2 khong phai
% thu pham -- nhung no van la MOT NUA cua co che. Gradient L2 = lambda*w KHONG
% DOI, trong khi gradient TD ti le voi reward. O thang 0.47/buoc thi TD ap dao
% L2; o thang 0.0022/buoc thi khong con ap dao nua. Cung mot lambda, hai ket
% qua khac han -- vi thang phan thuong khac nhau 215 lan.
%
% VI SAO PHAN THUONG O DAY NHO BAT THUONG: no la HIEU control-variate
% (c_base - c_agent)*dt giua hai chinh sach gan nhau. Do la lua chon CO CHU DICH
% de giam phuong sai (xem muc "Phan thuong" o dau file). Tac dung phu khong ai
% tinh: no lam yeu tin hieu TD di hai-ba bac. Ban va nay giu NGUYEN cau truc
% control-variate va chi tra lai thang do.
%
% NHAN MOT HANG SO KHONG DOI CHINH SACH TOI UU. argmax_pi E[sum gamma^t * c*r]
% = argmax_pi c * E[sum gamma^t r] voi moi c > 0. Day la phep bien doi an toan
% nhat trong so moi thu da thu: khong dung toi gamma, khong dung toi kien truc,
% khong dung toi dinh nghia "tot".
%
% rScale = 100 dua return/episode ve ~880 va thang Q (o gamma=0.99) ve ~22 --
% cung bac voi ~47 cua ban da hoi tu.
%
% LUU Y: nhan TOAN BO reward KE CA -failPen, de ty le giua phat that bai va
% phan thuong dense khong doi. Neu chi nhan phan dense thi that bai se tro nen
% re di 100 lan -- mot thay doi MDP that su, khac han phep nhan vo hai nay.
% ===========================================================================
d.rScale    = 1;           % he so nhan phan thuong. 'state' da dung thang -> 1.
                           % Neu quay ve rewardMode='cv' thi dat 100 (xem tren).

% ===========================================================================
% BAN VA 9 -- rewardMode. Bang chung: HAM THUONG THAT cua ban da hoi tu, trich
% tu WRIP_FLC_Model_RL.slx (khoi "MATLAB Function2" / RWIP_RL_Reward_safer):
%
%     r_alpha = -100 * alpha^2
%     r_theta = -5   * theta^2
%     r_u     = -0.3 * u^2
%     r_alive = +0.5 * exp(-200*alpha^2)
%     r_fail  = -50  * (1 - exp(-(alpha/0.5)^4))
%
% BA KHAC BIET CAU TRUC so voi ban 'cv' cua du an nay:
%
% (I) PHAN THUONG LA HAM CUA TRANG THAI, KHONG CUA MOT BIEN AN.
%     Ban 'cv' dung r = (cBase(k) - cA)*dt, trong do cBase(k) lay tu quy dao BO
%     NEN chay song song, danh chi so theo BUOC THOI GIAN k. Quan sat cua agent
%     la [th, thd, ph, phd, th_ref, u_FLC] -- KHONG co k, KHONG co trang thai
%     bo nen. Nen cung mot quan sat co the ung voi cBase khac nhau tuy kich ban
%     va thoi diem: PHAN THUONG KHONG PHAI HAM CUA QUAN SAT.
%
%     Critic khong the dinh gia mot dai luong ma no khong quan sat duoc. Do
%     dung la thu da do duoc: r = 0.94 (phan he thong hoc duoc) nhung
%     slope ~ 0.45 va khong tien them (phan con lai la nhieu KHONG KHU DUOC).
%     Khong phai gamma, khong phai L2, khong phai thang do, khong phai thoi gian.
%
%     Phep tru control-variate VAN bao toan chinh sach toi uu (cBase khong phu
%     thuoc hanh dong) -- cho do ghi chu thiet ke noi dung. Nhung no danh doi:
%     giam phuong sai return theo episode, doi lay mot phan thuong khong do
%     duoc tu quan sat. Cai gia do chua bao gio duoc tinh.
%
% (II) PHAT BINH PHUONG, KHONG PHAI L1. Dang binh phuong co do doc TANG DAN khi
%      ra xa 0 va -> 0 khi tien ve 0; cong voi r_alive doc dung quanh 0, agent
%      luon co gradient ro rang. Do la cach ban da hoi tu giai bai toan "bo nen
%      da rat tot nen chi phi gan nhu khong doi" MA KHONG can tru baseline.
%
% (III) PHAT THAT BAI TRON, KHONG CAT EPISODE. r_fail tang muot khi alpha -> 0.5
%      thay vi mot cu -200 roi ket thuc. Nho vay moi episode deu du do dai,
%      critic khong bao gio gap trang thai terminal gia, va toan bo van de
%      IsDone terminal-vs-truncated (ban va 2 muc C) BIEN MAT.
%
% 'state' chuyen HINH DANG tren sang bien cua du an nay. Hop le vi th cua ta va
% alpha cua ho la CUNG mot dai luong vat ly (goc con lac, rad) voi cung dai lam
% viec (~0.05-0.20), va thetaFail = 0.5 TRUNG KHIT nguong 0.5 trong r_fail cua ho.
% Phan chi phi dense giu NGUYEN cA da hieu chuan cua du an (L1, w_ph = 0.054 la
% trung binh nhan tren 6 kich ban), chi nhan 0.1 de o thang cua ho: tai th = 0.1
% thi -0.1*cA = -1.0, dung bang -100*0.1^2 = -1.0.
%
%   'state' : r = -0.1*cA + 0.5*exp(-200*th^2) - 50*(1-exp(-(th/thetaFail)^4))
%   'cv'    : r = (cBase - cA)*dt          (ban cu; can rScale = 100)
% ===========================================================================
d.rewardMode = 'state';    % 'state' (mac dinh, theo ban da hoi tu) | 'cv' (cu)
d.k_dense    = 0.1;        % he so dua cA ve thang cua ban da hoi tu
% ---------------------------------------------------------------------------
% k_alive = 0 (KHONG phai 0.5 nhu ban goc). Do duoc bang Test_Env_RewardDecomp:
%     r_alive = +1932.3 +- 38.6   |  tong return = +1178.2 +- 563.0
% Tuc no chiem 74% GIA TRI nhung chi 6% BIEN THIEN -- mot hang so cong them,
% khong phai tin hieu. Q se bi day len ~100*0.48 = 48 don vi hang so ma khong
% mang thong tin nao, lam giam do phan giai tuong doi cho phan thay doi that.
%
% VI SAO BAN GOC CAN NO MA BAN NAY THI KHONG: dense cua ban goc la BINH PHUONG
% (-100*alpha^2), co gradient TRIET TIEU khi alpha -> 0; r_alive ton tai chinh
% de vá cho do. Dense cua du an nay la L1 (|th|/th_s), gradient KHONG DOI ke ca
% sat 0. Chuyen r_alive sang day la chuyen mot lieu thuoc cho mot benh khong co.
%
% NEU sau nay doi dense sang binh phuong thi PHAI bat lai k_alive = 0.5 --
% luc do no khong con thua.
% ---------------------------------------------------------------------------
d.k_alive    = 0;          % TAT (xem tren). Ban goc dung 0.5 voi dense binh phuong.
d.a_alive    = 200;        % do rong thuong tron (chi co tac dung khi k_alive > 0)

% ===========================================================================
% BAN VA 10 -- k_fail = 0 VA GIU KET THUC SOM. Sua sai lam cua ban va 9.
% ---------------------------------------------------------------------------
% Ban va 9 bo ket thuc som (|th| > thetaFail) va thay bang phat TRON r_fail,
% de giong ban da hoi tu. DO LA MOT SAI LAM, va do duoc trong 200 episode:
%
%     Episode 100-200: reward = -45800 +- 100, steps = 1000-1300
%     Phan tich: r_fail ~ -38..-50/buoc x ~1200 buoc = -45600. KHOP.
%     => agent NAM YEN voi con lac da do, giua 0.5 va 1.5 rad, den khi vuot
%        1.5 thi episode moi ket thuc.
%
% VONG XOAY TU CHET: moi episode bom ~1200 buoc trang thai DA DO vao buffer.
% Sau ~100 episode, buffer gan nhu chi con trang thai da do, va agent khong
% con du lieu nao ve che do binh thuong de hoc lai. Khong the thoat.
%
% VI SAO BAN DA HOI TU KHONG DINH: no KHONG BAO GIO do. Du_RL cua no la +-2 V
% tren bus 12 V (17%); mot actor ngau nhien khong du tham quyen de lan at bo
% nen. Cua du an nay la +-8 V tren bus 24 V (33%) -- du de lat con lac ngay
% trong nhung episode tham do dau tien. Toi chuyen r_fail sang ma khong chuyen
% dieu kien lam cho no an toan.
%
% VA MOT LOI THU HAI, tinh dan: neu GIU CA HAI (ket thuc som + r_fail tron) thi
% agent co the chu dong lai con lac qua thetaFail de THOAT khoi dong -50/buoc.
% Ket thuc som tro thanh phan thuong. Hai co che nay loai tru nhau; phai chon.
%
% CHON: ket thuc som + phat mot lan (thiet ke goc cua du an nay). Ly do: no
% CHAN dinh thiet hai va giu buffer da dang, khong phu thuoc vao gia dinh
% "agent khong bao gio do" ma ta khong bao dam duoc voi bien +-8 V.
% ===========================================================================
d.k_fail    = 0;           % TAT phat tron. Ket thuc som + failPen dam nhiem.

% ===========================================================================
% failPen = chi_phi_tai_nguong / (1 - gamma).  SUA LOI DON VI 09/08.
% ---------------------------------------------------------------------------
% Ban dau toi dat failPen = k_dense*(thetaFail/th_s)*nStep -- tuc chi phi TICH
% LUY CA EPISODE (3000 buoc). Do la loi don vi: failPen la phan thuong MOT BUOC
% (terminal), nen no phai bang GIA TRI CHIET KHAU cua viec o lai nguong, tuc
% nhan voi CHAN TROI 1/(1-gamma) = 100 buoc, KHONG phai nhan voi nStep.
%
%     sai : 0.1*(0.5/0.010)*3000 = 15 000
%     dung: 0.1*(0.5/0.010)/0.01 =    500      -> lech 30 lan
%
% HAU QUA DO DUOC (300 episode, khoi 1):
%   phan thuong tung buoc binh thuong : -0.07 .. -6
%   muc tieu terminal -15 000          : lon gap 2500 lan gia tri lon nhat
%   -> voi MSE, MOT mau nhu vay trong minibatch 128 ap dao toan bo gradient
%      critic. Bang chung: Q0 truot deu tu -0.02 xuong -30.54, va ty le that
%      bai tang 4/10 -> 6/10 -> 10/10 trong khi cac episode KHONG do lai dang
%      TOT LEN (-914 -> -440 so voi moc bo nen -1132).
%   Hai xu huong nguoc nhau do la chu ky: critic bi mau ngoai lai keo lech ->
%   actor di theo gradient hong -> them that bai -> them mau ngoai lai.
%
% Voi failPen = 500: that bai o buoc 800 ton -800 - 500 = -1300 so voi ket thuc
% tot ~-300, tuc van phat gap ~4 lan, nhung khong con la mau ngoai lai so hoc
% (500 so voi chi phi/buoc 5 = 100 lan, DUNG bang chan troi -- dung thang do).
% ===========================================================================
d.gammaRef    = 0.99;      % gamma dung de suy failPen; PHAI khop DiscountFactor
d.failPenAuto = true;      % false -> dung d.failPen co dinh
d.k_fail     = 50;         % phat tron khi mat on dinh   (nguyen van)

% Phan thuong -- thanh phan dense (control-variate)
% ===========================================================================
% BAN VA 11 -- CHAN SO HANG phi TRONG CHI PHI. Do duoc, 09/08.
% ---------------------------------------------------------------------------
% cA = |th|/th_s + w_ph*|ph|/ph_s + w_u*(...)^2 + w_du*(...)^2
%
% So hang th BI CHAN tu nhien: episode ket thuc khi |th| > thetaFail = 0.5,
% nen no khong bao gio vuot 0.5/0.010 = 50.
% So hang ph KHONG BI CHAN GI CA: banh da quay tu do, va khi agent lam mat on
% dinh no co the dat hang tram rad.
%
% DO DUOC (khoi 1, lan chay 09/08 sau khi sua failPen):
%   reward TB = -3502 voi 23% that bai | moc bo nen = -1132
%   tach ra  : episode khong do ~-440 (TOT HON bo nen), episode do ~-13,750
%   mot episode do: phan dense ~-13,250 tren ~1000 buoc = -13.3/buoc
%   -> cA ~ 133, trong khi so hang th toi da chi la 50
%   -> phan con lai (~83) den tu phi: |phi| ~ 153 rad = 24 vong
%
% Do la DUNG co che ma failPen = 15000 tung gay ra, chi khac la no khong nam o
% mot mau terminal ma sinh ra o MOI BUOC: mot muc tieu TD lon gap ~35 lan gia
% tri binh thuong, lap lai hang nghin lan trong buffer. Critic bi keo lech,
% actor di theo, them that bai, them mau ngoai lai.
%
% CHAN o 25 rad: p99 cua |phi| do duoc tren bo nen la 18.9-25 rad tuy kich ban
% (muc [3c] cua Test_Env_RewardDecomp_v13). Tren muc do la che do banh da quay
% loan, khong con la van hanh binh thuong -- phan biet them giua 100 rad va 200
% rad khong mang thong tin dieu khien nao, chi bom nhieu vao critic.
%
% Sau khi chan: cA toi da = 50 (th) + 0.54*25 (ph) + ~0.07 = 63.5
%               -> phan thuong/buoc bi chan o -6.35, so voi -0.38 binh thuong.
%               Ty le 17 lan -- chap nhan duoc, thay vi 35 lan nhu truoc.
% ===========================================================================
d.phiCap = 25;             % rad, chan |phi| TRONG CHI PHI (khong chan dong hoc)

d.th_s = 0.010;  d.ph_s = 0.100;
d.w_ph = 0.054;  d.w_u  = 0.05;  d.w_du = 0.02;

% ===========================================================================
% Phan thuong -- thanh phan thua. r_bonus = 0 SAU BAN VA 4 (do duoc, 08/08).
% ---------------------------------------------------------------------------
% Ban va (B) them bonus voi ly do ghi ro: "gan diem hoi tu, chenh lech dense
% giua hai chinh sach gan-toi-uu tien ve 0, tin hieu thua bien do lon moi con
% phan biet duoc tot voi rat tot."
%
% Tien de do KHONG DUNG o diem lam viec hien tai. Do bang
% Test_Env_RewardDecomp_v13 tren chinh sach BC-only, dung phan bo kich ban
% huan luyen (p_kick = 1.0):
%
%     dense (cB - cA)*dt = +13.002 +- 18.798 moi episode
%
% Khong he tien ve 0 -- va no khop voi con so +15.5 +- 17.6 ma thiet ke GOC da
% do va da dung de bien minh cho ca kien truc ("60% episode khong chua tin hieu
% hoc", muc 2 dau Train_RL_v13_bc.m). Tuc phan dense mot minh DA LA tin hieu
% hoc duoc, khong can cuu tro.
%
% Con bonus thi do sai bien. Do duoc tren bo nen thuan, kich ban nhieu:
%
%     % thoi gian |theta| <= tightTh (0.010)  : 73.6%
%     % thoi gian |phi|   <= tightPh (0.050)  : 27.8%
%     % thoi gian thoa CA HAI (inTight)       : 27.5%
%     max|phi| quan sat duoc                  : 23.985 rad  (= 480 x tightPh)
%
% inTight kem dieu kien |phi| rieng dung 0.3 diem phan tram. Nghia la dieu kien
% |theta| -- MUC TIEU DIEU KHIEN THAT -- gan nhu khong dong gop gi, va bonus
% tren thuc te chi do "banh da co gan goc 0 khong". Voi 86% return den tu so
% hang nay, agent se danh 3.5 gio de toi uu vi tri goc banh da.
%
% tightPh = ph_s/2 duoc chon cho DOI XUNG voi th_s, khong tu vat ly. Bien do
% that cua phi lon hon 480 lan. Do khong phai tham so can tinh chinh -- do la
% dinh nghia cua tu "tot" bi dat sai bien.
%
% GIU LAI CO CHE (bonusBase van duoc tinh, phep tru van chay, muc [1] cua
% Test_Env_RewardDecomp van kiem tra duoc). Bat lai chi can doi mot so. Neu
% sau nay dense THUC SU phang ra gan diem hoi tu, hay bat lai bang cach:
%   (a) bo |phi| khoi inTight -- chi giu |theta| <= tightTh, va
%   (b) dat r_bonus ~ 0.5-1.0 (KHONG phai 10) de cung bac voi dense.
% Dung bat lai r_bonus = 10 voi inTight nhu cu.
%
% r_penalty GIU NGUYEN = 5: do duoc dong gop 0.0% tong va khong bao gio kich
% tren bo nen, nen no khong nuot gi. No chi kich khi AGENT phan ky ma bo nen
% thi khong -- mot tin hieu hiem, dung huong, phuong sai gan nhu bang khong.
% ===========================================================================
d.r_bonus       = 0;       % TAT (xem tren). Bat lai: doc ky muc (a)+(b) trong ghi chu.
d.r_penalty     = 5;       % phat MOT LAN khi vua vuot nguong phan ky (canh bao som)
d.tightTh       = 0.010;   % rad, nguong |theta| de tinh la "bam sat" (= th_s)
d.tightPh       = 0.050;   % rad, nguong |phi| de tinh la "bam sat" (= ph_s/2)
d.tightSustainS = 1.0;     % s, thoi gian lien tuc can de duoc thuong
d.divergeTh     = 0.25;    % rad, nguong |theta| coi la phan ky (< thetaFail=0.5)

% Kich ban huan luyen (tong = 1)
d.p_ic = 0.40;  d.p_kick = 0.40;  d.p_param = 0.20;
d.ic_theta   = [-0.20 0.20];
d.ic_thetad  = [-0.30 0.30];
d.ic_phid    = [-2.0  2.0 ];
d.kick_t     = [3 10];
d.kick_amp   = [0.5 3.0];
d.param_rel  = [0.8 1.2];      % huan luyen +-20%, DANH GIA +-30% (ngoai suy)

% So lan lay mau lai kich ban khi BO NEN khong song sot tren chinh nguong
% thetaFail cua agent. Xem ghi chu trong resetFcn.
d.maxResample = 20;

d.fisFile = 'fis_rwip_v13_best.fis';
d.verbose = false;

% Ban va 10: failPen suy ra tu cfg (xem ghi chu tren). Tinh SAU khi da hop nhat
% cfg nguoi dung, va chi khi nguoi dung KHONG tu dat failPen.
fn = fieldnames(d);
userSetFailPen = isfield(cfg,'failPen');
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}), cfg.(fn{k}) = d.(fn{k}); end
end
if cfg.failPenAuto && ~userSetFailPen && strcmpi(cfg.rewardMode,'state')
    % ---------------------------------------------------------------------
    % failPen = gia tri chiet khau cua "o lai trang thai TE NHAT duoc phep".
    % Phai dung cA DAY DU tai bien, khong chi so hang theta. Ban truoc chi lay
    % theta (= 50) nen ra 500, va kiem thu bang mo phong cho thay tai bien:
    %       V(tiep tuc) = -6.26*100 = -626   <   V(do) = -500
    % tuc AGENT CO LOI KHI CO TINH LAM DO -- dung loai lo hong dat nhat, vi
    % agent se hoc chinh xac cai do va ta se doc thanh "khong hoi tu".
    %
    % cA tai bien = thetaFail/th_s + w_ph*phiCap/ph_s + w_u + w_du
    %             = 50 + 13.5 + 0.05 + 0.02 = 63.57
    % failPen     = k_dense * 63.57 / (1 - gamma) = 0.1*63.57*100 = 636
    %
    % Tai DUNG bien, hai lua chon bang nhau (diem trung tinh). O moi trang thai
    % nam trong bien, chi phi nho hon nen tiep tuc luon co gia tri cao hon.
    % Khong can them he so an toan tuy y: bien la diem trung tinh theo dinh
    % nghia, va trong thuc te quy dao khong o lai bien mai nen tiep tuc con
    % tot hon nua so voi can duoi nay.
    % ---------------------------------------------------------------------
    cA_max = cfg.thetaFail/cfg.th_s + cfg.w_ph*cfg.phiCap/cfg.ph_s ...
             + cfg.w_u + cfg.w_du;
    cfg.failPen = cfg.k_dense * cA_max / (1 - cfg.gammaRef);
end

%% ---------------- Tham so doi tuong / bo nen ----------------
P = make_P_v13(cfg.fisFile);
assert(isfile(P.fisFile), 'Thieu %s -- chay pso_rwip_v13 truoc.', P.fisFile);
P.g_theta_nom = -(P.Kt/P.Ra)/P.Ip_eq;   % bo dieu khien GIU gia tri danh nghia
cfg.P = P;

%% ---------------- Dac ta khong gian ----------------
obsInfo = rlNumericSpec([6 1], 'Name','rwip_obs', ...
    'Description','[theta thetadot phi phidot theta_ref u_FLC] da chuan hoa');
actInfo = rlNumericSpec([1 1], 'LowerLimit',-1, 'UpperLimit',1, 'Name','u_corr_norm');

env = rlFunctionEnv(obsInfo, actInfo, ...
        @(a, ls) stepFcn(a, ls, cfg), @() resetFcn(cfg));
end


%% =====================================================================
function [obs, ls] = resetFcn(cfg)
% ===========================================================================
% LAY MAU KICH BAN CHO DEN KHI BO NEN SONG SOT TREN CHINH NGUONG CUA AGENT
% ---------------------------------------------------------------------------
% Hai loi da duoc sua o day (xem ghi chu review truoc khi chay):
%
% (4) NGUONG THAT BAI LECH GIUA AGENT VA BASELINE.
%     stepFcn cat episode khi |theta| > cfg.thetaFail = 0.5 va phat -failPen.
%     Nhung cBase truoc day sinh tu sim_rwip_v13, von chi coi la phan ky khi
%     |theta| > 1.5 -- do la nguong PHAN KY SO HOC, khong phai nguong THAT BAI
%     DIEU KHIEN. Ket qua: ton tai episode ma bo nen duoc chay toi theta = 0.8
%     rad va tich chi phi binh thuong, con agent bi cat o 0.5 kem -200. Ham
%     thuong khi do thien vi CHONG LAI agent o dung nhom episode kho -- noi
%     duy nhat co du dia. Nay bo nen phai thoa CUNG mot nguong.
%
% (5) BOM PHUONG SAI KHI BO NEN HONG.
%     Ban cu: bo nen hong -> cBase(:) = thetaFail/th_s = 50 phang. Mot episode
%     nhu vay cho thuong ~ (50 - cA)*0.01*4000 ~ +2000, so voi dai binh thuong
%     +15.5 +- 17.6. MOT episode du lam lech critic. Nay khong con nhanh do:
%     kich ban nao bo nen khong song sot thi LAY MAU LAI, nen cBase luon la
%     chi phi cua mot quy dao ma agent cung duoc phep di het.
%
% Tan suat du kien: thap. Do duoc o Bang 4, dinh |theta| cua bo nen chi ~0.043
% rad tai xung 2.0 rad/s, nen nguong 0.5 hiem khi cham. Sua nay nham vao dung
% loai su kien hiem-ma-tai-hai, khong phai loai thuong gap.
% ===========================================================================
ok = false;
for iTry = 1:max(1, cfg.maxResample)
    ls = drawScenario(cfg, false);
    [ok, cBase, bonusBase, penBase] = baseRollout(ls, cfg);
    if ok, break; end
end
if ~ok
    % Khong lay duoc kich ban kha thi sau maxResample lan: lui ve kich ban nhe
    % thay vi bia mot cBase gia. Canh bao de biet ngay, khong am tham.
    warning('rwip_env_v13:resample', ...
        ['Bo nen khong song sot %d lan lien tiep -- dung kich ban du phong. ' ...
         'Neu canh bao nay lap lai, xem lai cfg.kick_amp / cfg.ic_*.'], cfg.maxResample);
    ls = drawScenario(cfg, true);
    [~, cBase, bonusBase, penBase] = baseRollout(ls, cfg);
end
ls.cBase     = cBase;
ls.bonusBase = bonusBase;   % ban va 2 muc D -- tham chieu de tru bonus
ls.penBase   = penBase;     % ban va 2 muc D -- tham chieu de tru penalty
ls.nTry      = iTry;

% --- Trang thai chay ---
ls.x = ls.x0;  ls.k = 0;  ls.t = 0;  ls.uPrev = 0;  ls.kicked = ~isfinite(ls.kickT);
ls.uMaxSeen = 0;  ls.uBaseMaxSeen = eps;
ls.tightCounter = 0;  ls.divergeFlag = false;   % trang thai cho phan thuong thua

% --- Cong don phan ra reward (ban va 2 muc E) -- CHI de chan doan ---
ls.sumDense = 0;   % tong (cB - cA)*dt
ls.sumBonus = 0;   % tong (bonus - bonusBase)  <- da tru baseline
ls.sumPen   = 0;   % tong (penalty - penBase)  <- da tru baseline
ls.nBonus   = 0;   % so lan agent kich bonus (chua tru), de doi chieu voi log cu

[obs, ~] = observe(ls, cfg);
end


%% =====================================================================
function ls = drawScenario(cfg, mild)
%DRAWSCENARIO  Boc mot kich ban huan luyen. mild = true -> ban nhe (du phong).
ls = struct();
ls.P = cfg.P;  ls.kickT = inf;  ls.kickA = 0;
r = rand();
if r < cfg.p_ic
    ls.scen = 'ic';
elseif r < cfg.p_ic + cfg.p_kick
    ls.scen = 'kick';
    ls.kickT = urand(cfg.kick_t);
    amp = cfg.kick_amp;
    if mild, amp = [amp(1), amp(1) + 0.25*(amp(2)-amp(1))]; end
    ls.kickA = urand(amp) * sign(rand-0.5);
else
    ls.scen = 'param';
    for f = {'mgl_eq','Ip_eq','Kt'}
        ls.P.(f{1}) = cfg.P.(f{1}) * urand(cfg.param_rel);
    end
    % Bo dieu khien KHONG biet doi tuong da doi -> giu g_theta danh nghia
    ls.P.g_theta_nom = cfg.P.g_theta_nom;
end

icT = cfg.ic_theta;  icTd = cfg.ic_thetad;  icPd = cfg.ic_phid;
if mild, icT = 0.25*icT;  icTd = 0.25*icTd;  icPd = 0.25*icPd; end
ls.x0 = [ urand(icT); urand(icTd); 0; urand(icPd) ];
end


%% =====================================================================
function [ok, cBase, bonusBase, penBase] = baseRollout(ls, cfg)
%BASEROLLOUT  Quy dao BO NEN chay song song (phan thuong control-variate).
%   ok = false neu bo nen khong song sot tren nguong cua AGENT (thetaFail),
%   khong phai nguong phan ky so hoc 1.5 rad cua sim_rwip_v13.
%
%   Tra ve BA chuoi tham chieu, deu lay tren CUNG mot quy dao bo nen:
%     cBase(k)     -- chi phi dense (nhu truoc)
%     bonusBase(k) -- bonus ma BO NEN se nhan tai buoc k (ban va 2, muc D)
%     penBase(k)   -- penalty ma BO NEN se chiu tai buoc k
%   Ba chuoi nay KHONG phu thuoc hanh dong cua agent -> tru di la control
%   variate hop le: giam phuong sai, khong doi chinh sach toi uu.
%
%   LUU Y ve so hang u trong cA: cA co them w_u*(u_corr/uMax)^2 va w_du*(...)^2
%   ma cB khong co (bo nen khong co u_corr -- theo dinh nghia u_corr=0). Do la
%   CO Y: hai so hang do la chi phi RIENG cua lop hieu chinh, phai bi phat that
%   chu khong duoc trieu tieu. Chi cac so hang TRANG THAI (theta, phi) moi duoc
%   ghep cap de tru.
Pb = ls.P;
if isfinite(ls.kickT), Pb.dist = struct('t', ls.kickT, 'dthetadot', ls.kickA); end
ob = sim_rwip_v13('PSOFLC', Pb, ls.x0, cfg.Tf, cfg.Ts);

nA        = round(cfg.Tf/cfg.Ts_agent);
cBase     = zeros(nA,1);
bonusBase = zeros(nA,1);
penBase   = zeros(nA,1);
ok        = ~ob.diverged && max(abs(ob.theta)) <= cfg.thetaFail;
if ~ok, return; end

step         = round(cfg.Ts_agent/cfg.Ts);
sustainSteps = round(cfg.tightSustainS / cfg.Ts_agent);
tightCnt     = 0;      % guong CHINH XAC logic dem trong stepFcn
divFlag      = false;  % guong CHINH XAC logic co phan ky trong stepFcn
for k = 1:nA
    i  = min(k*step, numel(ob.t));
    th = ob.theta(i);  ph = ob.phi(i);
    % Ban va 11: chan giong het cA de hai ben van so sanh duoc (che do 'cv').
    cBase(k) = abs(th)/cfg.th_s + cfg.w_ph*min(abs(ph),cfg.phiCap)/cfg.ph_s;

    % --- bonus cua bo nen (cung nguong, cung quy tac "moi 1 s lien tuc") ---
    if abs(th) <= cfg.tightTh && abs(ph) <= cfg.tightPh
        tightCnt = tightCnt + 1;
    else
        tightCnt = 0;
    end
    if tightCnt > 0 && mod(tightCnt, sustainSteps) == 0
        bonusBase(k) = cfg.r_bonus;
    end

    % --- penalty cua bo nen (phat MOT LAN moi lan vua vuot nguong) ---
    if abs(th) > cfg.divergeTh
        if ~divFlag, penBase(k) = cfg.r_penalty;  divFlag = true; end
    else
        divFlag = false;
    end
end
end


%% =====================================================================
function [obs, reward, isdone, ls] = stepFcn(action, ls, cfg)
P = ls.P;  Ts = cfg.Ts;
nSub = round(cfg.Ts_agent/Ts);
u_corr = cfg.uMax * max(-1, min(1, action(1)));

for s = 1:nSub
    [uB, tref] = baseCmd(ls.x, P);
    V = max(-24, min(24, uB + u_corr));
    ls.uMaxSeen     = max(ls.uMaxSeen, abs(u_corr));
    ls.uBaseMaxSeen = max(ls.uBaseMaxSeen, abs(uB));

    f  = @(X) plantDeriv(X, V, P);
    k1 = f(ls.x); k2 = f(ls.x+Ts/2*k1); k3 = f(ls.x+Ts/2*k2); k4 = f(ls.x+Ts*k3);
    ls.x = ls.x + Ts/6*(k1+2*k2+2*k3+k4);
    ls.t = ls.t + Ts;

    if ~ls.kicked && ls.t >= ls.kickT
        ls.x(2) = ls.x(2) + ls.kickA;  ls.kicked = true;
    end
    if ~all(isfinite(ls.x)) || abs(ls.x(1)) > cfg.thetaFail, break; end
end
ls.k = ls.k + 1;

% --- Chi phi agent va phan thuong ---
th = ls.x(1); ph = ls.x(3);
failed = ~all(isfinite(ls.x)) || abs(th) > cfg.thetaFail;
% BAN VA 10: da BO doan cua ban va 9 tung noi rong dieu kien that bai len 1.5
% rad o che do 'state'. Do la nguyen nhan vong xoay tu chet -45800 (xem ghi chu
% dai o phan cau hinh). Ca hai che do gio dung CUNG mot dieu kien: |th| >
% thetaFail thi ket thuc episode NGAY, khong de agent nam lai an -50/buoc.
if failed
    ls.sumPen = ls.sumPen + cfg.failPen*cfg.rScale;   % chan doan (muc E)
    rl_rho_log('add', ls, true);   % chan doan, KHONG tham gia phan thuong
    reward = -cfg.failPen * cfg.rScale;   % ban va 7: nhan CA phat that bai
    isdone = 1;   % TERMINAL that su (trang thai hap thu) -- doi lap voi
                  % truong hop het gio o cuoi ham, xem BAN VA 2 (C).
    obs = zeros(6,1);  return
end

% Ban va 11: |ph| bi CHAN o phiCap TRONG CHI PHI. Chi so hang chi phi bi chan
% -- dong hoc va quan sat giu nguyen |ph| that. Xem ghi chu dai o cho dinh
% nghia d.phiCap ve ly do (so hang nay la nguon mau ngoai lai duy nhat con lai
% sau khi da sua failPen).
cA = abs(th)/cfg.th_s + cfg.w_ph*min(abs(ph),cfg.phiCap)/cfg.ph_s ...
   + cfg.w_u*(u_corr/cfg.uMax)^2 + cfg.w_du*((u_corr-ls.uPrev)/cfg.uMax)^2;
cB = ls.cBase(min(ls.k, numel(ls.cBase)));

% --- So hang thua (bonus/penalty nguong) -- xem "BAN VA" muc (B) va (D) ---
inTight = abs(th) <= cfg.tightTh && abs(ph) <= cfg.tightPh;
if inTight, ls.tightCounter = ls.tightCounter + 1; else, ls.tightCounter = 0; end
sustainSteps = round(cfg.tightSustainS / cfg.Ts_agent);
bonus = 0;
if ls.tightCounter > 0 && mod(ls.tightCounter, sustainSteps) == 0
    bonus = cfg.r_bonus;         % kich lai moi khi tron them 1s bam sat lien tuc
end
penalty = 0;
isDiverging = abs(th) > cfg.divergeTh;
if isDiverging && ~ls.divergeFlag
    penalty = cfg.r_penalty;     % phat MOT LAN tai thoi diem vua vuot nguong
    ls.divergeFlag = true;
elseif ~isDiverging
    ls.divergeFlag = false;      % dat lai de co the phat lai neu tai phan ky
end

% --- BAN VA 2 (D): tru tham chieu BO NEN khoi ca hai so hang thua ---------
% Truoc ban va nay, bonus vao thang phan thuong -> ~90-95% return la mot xo so
% kich ban ma agent khong dieu khien duoc. Sau khi tru, con lai dung phan
% "agent bam sat duoc bao nhieu lau HON bo nen tren CUNG kich ban".
kb        = min(ls.k, numel(ls.bonusBase));
% Ban va 7: nhan rScale vao TUNG so hang, de bang phan ra chan doan (muc E)
% van cong dung bang reward that -- neu chi nhan o dong tong thi log se lech
% 100 lan so voi reward agent nhan, va lan chan doan sau se doc nham.
if strcmpi(cfg.rewardMode,'state')
    % --- BAN VA 9: ham cua TRANG THAI, khong dung cBase --------------------
    % Chuyen nguyen hinh dang tu RWIP_RL_Reward_safer cua ban da hoi tu.
    % cB van duoc tinh o tren nhung KHONG vao phan thuong -- giu lai de bang
    % chan doan (muc E) van so sanh duoc voi bo nen.
    rDense   = -cfg.k_dense * cA;
    rAlive   =  cfg.k_alive * exp(-cfg.a_alive * th^2);
    rFailSm  = -cfg.k_fail  * (1 - exp(-(th/cfg.thetaFail)^4));
    dBonus   =  rAlive  * cfg.rScale;    % ghi vao cot "bonus" cua log
    dPenalty = -rFailSm * cfg.rScale;    % ghi vao cot "pen"  (dau: pen bi TRU)
    rDense   =  rDense  * cfg.rScale;
else
    dBonus    = (bonus   - ls.bonusBase(kb)) * cfg.rScale;
    dPenalty  = (penalty - ls.penBase(kb))   * cfg.rScale;
    rDense    = (cB - cA) * cfg.Ts_agent     * cfg.rScale;
end

reward   = rDense + dBonus - dPenalty;
ls.uPrev = u_corr;

% --- Chan doan: cong don phan ra reward (muc E), KHONG tham gia phan thuong --
ls.sumDense = ls.sumDense + rDense;
ls.sumBonus = ls.sumBonus + dBonus;
ls.sumPen   = ls.sumPen   + dPenalty;
if bonus > 0, ls.nBonus = ls.nBonus + 1; end

% ===========================================================================
% BAN VA 2 (C): HET GIO LA TRUNCATION, KHONG PHAI TERMINAL.
% ---------------------------------------------------------------------------
%   IsDone = 1 -> RL Toolbox coi la trang thai TERMINAL, target = r
%   IsDone = 2 -> coi la TRUNCATED (het ngan sach buoc), target = r + gamma*Q'
% Tai t = Tf con lac VAN DANG DUNG -- day la cat ngang vi het gio mo phong,
% khong phai ket cuc that. Tra ve 1 (nhu ban cu) day critic rang gia tri tai
% cuoi episode dung bang mot buoc reward le (~0.008), va vi 140/140 episode
% ket thuc bang duong nay, 100% du lieu dinh loi.
%
% Doi voi nhanh THAT BAI o tren (|theta| > thetaFail) thi 1 la DUNG: do la
% trang thai hap thu that, khong con gia tri tuong lai de bootstrap.
%
% PHONG THU: mot so phien ban RL Toolbox cu chi chap nhan logical. Neu phien
% ban cua ban khong hieu gia tri 2, no se bi ep ve true = 1 va ta quay lai
% dung hanh vi cu (khong te hon ban cu), nhung precheck_rl_v13 se bao.
% ===========================================================================
if ls.k >= numel(ls.cBase)
    isdone = 2;                    % TRUNCATED -- van bootstrap
    rl_rho_log('add', ls, false);  % chan doan, KHONG tham gia phan thuong
else
    isdone = 0;
end

[obs, ~] = observe(ls, cfg);
end


%% =====================================================================
function [obs, tref] = observe(ls, cfg) %#ok<INUSD>
[uB, tref] = baseCmd(ls.x, ls.P);
obs = rl_obs_v13(ls.x(1), ls.x(2), ls.x(3), ls.x(4), tref, uB);
end

function [uB, tref] = baseCmd(x, P)
persistent SURF GRD KEY
key = P.fisFile;
if isempty(KEY) || ~strcmp(KEY, key)
    fis = readfis(P.fisFile);
    GRD = linspace(-1,1,201);
    [A,B] = ndgrid(GRD,GRD);
    SURF = reshape(evalfis(fis,[A(:) B(:)]), 201, 201);
    KEY  = key;
end
th=x(1); thd=x(2); ph=x(3); phd=x(4);
tref = max(-P.theta_ref_max, min(P.theta_ref_max, -(P.Kp_phi_mag*ph + P.Kv_phi_mag*phd)));
i1 = max(-1,min(1, P.Ke *(th - tref)));
i2 = max(-1,min(1, P.Kde*thd));
uB = P.Ku*bilin(GRD, SURF, i1, i2) + P.Kd_theta*thd + P.Kd_phi*phd;
end

function dx = plantDeriv(x, V, P)
th=x(1); thd=x(2); phd=x(4);
tau  = (P.Kt/P.Ra)*(V - P.Ke_m*phd);
thdd = (-tau + P.br*phd + P.mgl_eq*sin(th) - P.bp*thd)/P.Ip_eq;
dx = [thd; thdd; phd; (tau - P.br*phd)/P.Ir - thdd];
end

function v = bilin(g, S, a, b)
n = numel(g); h = 2/(n-1);
xa = (max(-1,min(1,a))+1)/h;  xb = (max(-1,min(1,b))+1)/h;
i = min(max(floor(xa),0), n-2); j = min(max(floor(xb),0), n-2);
fa = xa-i; fb = xb-j; i=i+1; j=j+1;
v = S(i,j)*(1-fa)*(1-fb) + S(i+1,j)*fa*(1-fb) ...
  + S(i,j+1)*(1-fa)*fb   + S(i+1,j+1)*fa*fb;
end

function v = urand(r), v = r(1) + (r(2)-r(1))*rand(); end
