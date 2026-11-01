% Round 5 - functional smoke tests (read-only; nothing saved except this log)
here = fileparts(mfilename('fullpath')); proj = fileparts(fileparts(fileparts(here)));
diary(fullfile(here,'..','logs',['smoke_test_' strrep(version('-release'),' ','') '.txt'])); diary on
fprintf('Date: %s | %s\n', datestr(now,31), version);
T = {};
T{end+1} = {'license test Simulink', @() license('test','Simulink')};
T{end+1} = {'license test Reinforcement_Learn', @() license('test','Reinforcement_Learn')};
T{end+1} = {'license test Fuzzy_Toolbox', @() license('test','Fuzzy_Toolbox')};
T{end+1} = {'license test Neural_Network_Toolbox', @() license('test','Neural_Network_Toolbox')};
T{end+1} = {'license test Control_Toolbox', @() license('test','Control_Toolbox')};
T{end+1} = {'license test GADS_Toolbox', @() license('test','GADS_Toolbox')};
T{end+1} = {'rlNumericSpec', @() size(rlNumericSpec([6 1]).LowerLimit)};
T{end+1} = {'rlTD3Agent exists', @() exist('rlTD3Agent')};
T{end+1} = {'mamfis (Fuzzy)', @() class(mamfis)};
T{end+1} = {'readfis fis_rwip_v13_best.fis', @() class(readfis(fullfile(proj,'fis_rwip_v13_best.fis')))};
T{end+1} = {'tf (Control)', @() class(tf(1,[1 1]))};
T{end+1} = {'dlnetwork (DeepLearning)', @() class(dlnetwork(featureInputLayer(2)))};
T{end+1} = {'load_system RWIP_PSO_FLC', @() load_system(fullfile(proj,'RWIP_PSO_FLC.slx'))};
T{end+1} = {'bdIsLoaded RWIP_PSO_FLC', @() bdIsLoaded('RWIP_PSO_FLC')};
T{end+1} = {'particleswarm exists', @() exist('particleswarm')};
for i=1:numel(T)
  try, r = T{i}{2}(); if isobject(r)||iscell(r), r=class(r); end, fprintf('OK    %-40s -> %s\n', T{i}{1}, mat2str(r));
  catch ME, fprintf('FAIL  %-40s -> %s\n', T{i}{1}, strtok(ME.message, newline)); end
end
try, bdclose('all'); catch, end
fprintf('files: '); disp(exist(fullfile(proj,'fis_rwip_v13_best.fis'),'file'));
diary off
