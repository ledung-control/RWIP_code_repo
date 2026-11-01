function agent = r5_E1_train(seed, gamma, sd)
% Copy of the call in Train_RL_v13_repro.m (authors) with DiscountFactor = gamma instead of the hard-coded 0.995.
% All other options are those pinned in Train_RL_v13_repro.m (default branch: 60 BC ep, 3000 BC iters, TailFrac 0.15,
% 20 warm-up ep, 18 fine-tune ep, actor LR 1e-5, GradThresh Inf, SkipCriticGate/SkipB1Gate true, SkipBCGate false).
rng(seed,'twister');
cfg0 = struct('p_ic',0.0, 'p_kick',1.0, 'p_param',0.0, 'Tf',40, 'Ts',1e-3, 'Ts_agent',0.01, 'uMax',10, 'thetaFail',0.5, 'failPen',200, ...
              'th_s',0.010, 'ph_s',0.100, 'w_ph',0.054, 'w_u',0.05, 'w_du',0.02, 'kick_t',[3 10], 'kick_amp',[0.5 3.0], ...
              'rewardMode','cv', 'rScale',1, 'r_bonus',0, 'r_penalty',0, 'phiCap',inf, 'k_alive',0, 'k_fail',0, 'failPenAuto',false);
tOpts = struct('BCEpisodes',60,'BCIters',3000,'TailFrac',0.15,'WarmupEpisodes',20,'MaxEpisodes',18,'WarmupChunk',20,'ActorLRFineTune',1e-5, ...
    'DiscountFactor',gamma,'GradThresh',Inf,'RewardCfg',cfg0,'SkipCriticGate',true,'SkipB1Gate',true,'SkipBCGate',false,'SaveDir',sd,'Plots','none');
agent = Train_RL_v13_bc(seed, tOpts);
end
