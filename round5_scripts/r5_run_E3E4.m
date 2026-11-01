function r5_run_E3E4()
% sequential: E3 (rate sensitivity), E4 search (2 x 600 LQR samples), E4 compare.  All resumable / write CSV as they go.
cd(fileparts(mfilename('fullpath')));
r5_E3;
r5_E4('search');
r5_E4('compare');
end
