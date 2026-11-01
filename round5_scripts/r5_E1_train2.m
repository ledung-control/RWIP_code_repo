function agent = r5_E1_train2(seed, gamma, sd, skipBC)
% Same as r5_E1_train.m (copy of the Train_RL_v13_repro call, DiscountFactor = gamma) with SkipBCGate as a parameter.
% r5_E1_train.m is left untouched (it produced the 15 tuning runs).  skipBC=false -> identical to r5_E1_train.
rng(seed,'twister');
cfg0 = struct('p_ic',0.0, 'p_kick',1.0, 'p_param',0.0, 'Tf',40, 'Ts',1e-3, 'Ts_agent',0.01, 'uMax',10, 'thetaFail',0.5, 'failPen',200, ...
              'th_s',0.010, 'ph_s',0.100, 'w_ph',0.054, 'w_u',0.05, 'w_du',0.02, 'kick_t',[3 10], 'kick_amp',[0.5 3.0], ...
              'rewardMode','cv', 'rScale',1, 'r_bonus',0, 'r_penalty',0, 'phiCap',inf, 'k_alive',0, 'k_fail',0, 'failPenAuto',false);
tOpts = struct('BCEpisodes',60,'BCIters',3000,'TailFrac',0.15,'WarmupEpisodes',20,'MaxEpisodes',18,'WarmupChunk',20,'ActorLRFineTune',1e-5, ...
    'DiscountFactor',gamma,'GradThresh',Inf,'RewardCfg',cfg0,'SkipCriticGate',true,'SkipB1Gate',true,'SkipBCGate',skipBC,'SaveDir',sd,'Plots','none');
agent = Train_RL_v13_bc(seed, tOpts);
end
