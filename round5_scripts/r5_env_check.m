% Round 5 - environment check (read-only). Writes logs/env_check.txt
here = fileparts(mfilename('fullpath')); L = fullfile(here,'..','logs'); 
diary(fullfile(L,['env_check_' strrep(version('-release'),' ','') '.txt'])); diary on
fprintf('Date: %s\n', datestr(now,31));
fprintf('MATLAB: %s\n', version); disp(version('-release'));
fprintf('OS: %s\n', computer); try, fprintf('%s\n', system_dependent('getos')); catch, end
fprintf('Java: %s\n', version('-java'));
try, fprintf('numcores: %d\n', feature('numcores')); catch, end
fprintf('matlabroot: %s\n', matlabroot);
fprintf('\n--- ver (installed toolboxes) ---\n'); v = ver; for i=1:numel(v), fprintf('%s | %s | %s\n', v(i).Name, v(i).Version, v(i).Release); end
fprintf('\n--- license checkout ---\n');
feats = {'MATLAB','Simulink','Reinforcement_Learn','Neural_Network_Toolbox','Fuzzy_Toolbox','Control_Toolbox','Optimization_Toolbox','GADS_Toolbox','Statistics_Toolbox','Distrib_Computing_Toolbox','Signal_Toolbox','Simscape','MATLAB_Coder'};
for i=1:numel(feats), try, [ok,msg]=license('checkout',feats{i}); fprintf('%-34s test=%d checkout=%d %s\n',feats{i},license('test',feats{i}),ok,msg); catch ME, fprintf('%-34s err %s\n',feats{i},ME.message); end, end
fprintf('\n--- GPU/parallel ---\n'); try, fprintf('gpuDeviceCount: %d\n', gpuDeviceCount); catch ME, fprintf('gpu: %s\n', ME.message); end
fprintf('Java: %s\n', version('-java'));
fprintf('\n--- project folder ---\n'); proj = fileparts(fileparts(fileparts(here))); fprintf('project: %s\n', proj);
d = dir(fullfile(proj,'*.m')); for i=1:numel(d), fprintf('%s | %d | %s\n', d(i).name, d(i).bytes, d(i).date); end
d = dir(fullfile(proj,'*.slx')); for i=1:numel(d), fprintf('%s | %d | %s\n', d(i).name, d(i).bytes, d(i).date); end
d = dir(fullfile(proj,'*.mat')); for i=1:numel(d), fprintf('%s | %d | %s\n', d(i).name, d(i).bytes, d(i).date); end
diary off
