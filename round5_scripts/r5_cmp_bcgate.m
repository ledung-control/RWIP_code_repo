function r5_cmp_bcgate()
% Read-only check: are the three bcgate BC+TD3 agents (c4,c5,c6, seed 102) identical in actor weights? Also compares actor/critic with BC-only.
here = fileparts(mfilename('fullpath')); A = fullfile(here,'..','agents','E1_bcgate');
ids = {'c4_SPH40_g0.995_s102','c5_SPH40_g0.999_s102','c6_SPH40_g0.9995_s102'};
fid = fopen(fullfile(here,'..','results','E1_bcgate_actor_critic_compare.txt'),'w');
W = cell(3,1); Wc = cell(3,1); G = cell(3,1);
for i=1:3
  S = load(fullfile(A,ids{i},'agent_bc_seed102.mat')); ag = S.agent;
  W{i} = flat(getLearnableParameters(getActor(ag)));
  try, cr = getCritic(ag); v = []; for q=1:numel(cr), pq = getLearnableParameters(cr(q)); v = [v; flat(pq)]; end, Wc{i} = v; catch ME, Wc{i} = []; fprintf(fid,'critic read error: %s\n', ME.message); end
  try, G{i} = ag.AgentOptions.DiscountFactor; catch, G{i} = NaN; end
  fprintf(fid,'%s: nActor=%d nCritic=%d DiscountFactor=%g\n', ids{i}, numel(W{i}), numel(Wc{i}), G{i});
end
B = load(fullfile(A,ids{1},'agent_bconly_seed102.mat')); Wb = flat(getLearnableParameters(getActor(B.agent)));
fprintf(fid,'max|actor c4-c5|=%g  max|c4-c6|=%g  max|c5-c6|=%g\n', mx(W{1},W{2}), mx(W{1},W{3}), mx(W{2},W{3}));
fprintf(fid,'max|actor c4-BConly|=%g\n', mx(W{1},Wb));
if ~isempty(Wc{1}), fprintf(fid,'max|critic c4-c5|=%g  max|c4-c6|=%g\n', mx(Wc{1},Wc{2}), mx(Wc{1},Wc{3})); end
fclose(fid);
end
function v = flat(c), v = []; if ~iscell(c), c = {c}; end, for k=1:numel(c), x = c{k}; if iscell(x), v = [v; flat(x)]; else, v = [v; double(extractdata(x(:)))]; end, end, end
function m = mx(a,b), if numel(a)~=numel(b), m = NaN; else, m = max(abs(a-b)); end, end
