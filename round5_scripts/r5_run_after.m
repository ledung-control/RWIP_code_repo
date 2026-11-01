function r5_run_after()
% runs the critic comparison, then the E1 S1/S2 supplement
cd(fileparts(mfilename('fullpath')));
r5_cmp_bcgate;
r5_E1_S1S2;
end
