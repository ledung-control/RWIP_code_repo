function [V, z, s] = SMCESO_cascade(theta, theta_ref, z, g_theta, Ts, G)
%SMCESO_CASCADE  SMC + ESO bac 3, PHIEN BAN CO DIEU KHIEN BANH DA.
%
% ===========================================================================
% KHAC BIET SO VOI SMCESO_controller.m (ban goc trong Baseline_SMC_ESO/)
% ---------------------------------------------------------------------------
% Ban goc:      s = c*theta + omega_p
%   -> mat truot CHI chua theta va theta_dot. Bo dieu khien KHONG he dieu chinh
%      banh da. No on dinh con lac bang cach do toan bo mo-men dong luong vao
%      banh da roi de do. Bang chung tu chinh ket qua da luu cua ban
%      (Eval_PSOFLC_vs_SMCESO_results.mat, 60 s):
%           RMSEphi = 10.02 rad ,  IAEphi = 523.05
%      Banh da KHONG ve 0. Tren phan cung that no se chay den bao hoa toc do
%      va mat hoan toan kha nang dieu khien.
%
% Ban nay:      dung DUNG vong ngoai cascade cua PSO-FLC v13
%                 theta_ref = sat(-(Kp_phi*phi + Kv_phi*phi_dot), +-theta_ref_max)
%               roi SMC bam theta_ref:   s = c*e + edot,  e = theta - theta_ref
%   -> ca hai bo dieu khien chiu CHUNG mot vong ngoai, CHUNG mot doi tuong,
%      CHUNG mot dac ta. So sanh khi do co lap dung thu can so: luat vong trong
%      (mo so vs truot), khong lan khac biet kien truc.
%
% ---------------------------------------------------------------------------
% TAI SAO KHONG CAN theta_ref_dot
% ---------------------------------------------------------------------------
% ESO chay TREN BIEN SAI SO e (khong phai tren theta). Nho vay theta_ref_ddot
% tu dong nam trong "nhieu tong hop" z3 va duoc trieu tieu, khong phai vi phan
% theta_ref. Day la cach lam chuan cua ADRC.
%       z1 ~ e        z2 ~ edot        z3 ~ f_theta - theta_ref_ddot
%       s  = c*e + z2
%       V  = (1/g_theta)*( -c*z2 - z3 - k*s - eta*sat(s/phi_bl) )
%
% ESO bac 3, 3 cuc dat tai -omega0:  beta1=3*w0, beta2=3*w0^2, beta3=w0^3
%
% ---------------------------------------------------------------------------
% INPUT
%   theta      : goc con lac do duoc tai buoc hien tai (rad)
%   theta_ref  : lenh goc tu vong ngoai (rad), DA bao hoa
%   z          : [z1;z2;z3] trang thai ESO tu buoc truoc
%   g_theta    : he so control-affine = -(Kt/Ra)/Ip_eq  (AM, ~ -2.79)
%   Ts         : buoc thoi gian (s)
%   G          : struct gain, cac truong c, k, eta, phi_bl, w0, Vmax
% OUTPUT
%   V : dien ap lenh (V), da bao hoa +-G.Vmax
%   z : trang thai ESO cap nhat
%   s : gia tri mat truot (de ve do thi/go loi)
%
% GAIN DE XUAT (da do bang mo phong doc lap, 60 s, theta0=0.10):
%   c=8  k=5  eta=1.0  phi_bl=0.05  w0=20
%   voi vong ngoai Kp_phi=-0.12, Kv_phi=-0.018, theta_ref_max=0.035
%   -> IAEtheta=0.137  IAEphi=1.25  RMSEphi=0.098  vot_lo=0.0493
%      phi(60)=-2e-19  phi_dot(60)=-8e-19  Urms=0.189
%
% LUU Y w0: ban goc dung w0=60. Voi cascade, w0=60 gay chattering ma banh da
% TICH PHAN HAI LAN -> phi khong bao gio dung (do duoc Urms 7-12 V thay vi
% 0.19 V, phi_dot(60) con dao dong 0.2-3 rad/s). w0=20 khac phuc triet de.
% ===========================================================================

c      = G.c;
k      = G.k;
eta    = G.eta;
phi_bl = G.phi_bl;
w0     = G.w0;
if isfield(G,'Vmax'), Vmax = G.Vmax; else, Vmax = 24; end

beta1 = 3*w0;  beta2 = 3*w0^2;  beta3 = w0^3;

e = theta - theta_ref;

% --- Luat dieu khien (dung uoc luong ESO tu buoc truoc) ---
s     = c*e + z(2);
sat_s = max(-1, min(1, s/phi_bl));           % lop bien thay sign(s)
V     = (1/g_theta) * ( -c*z(2) - z(3) - k*s - eta*sat_s );
V     = max(-Vmax, min(Vmax, V));

% --- Cap nhat ESO (Euler thuan) tren bien sai so e ---
e1 = e - z(1);
z  = z + Ts*[ z(2) + beta1*e1 ;
              z(3) + g_theta*V + beta2*e1 ;
              beta3*e1 ];
end
