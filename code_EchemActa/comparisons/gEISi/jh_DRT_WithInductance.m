function Zhat=jh_DRT_WithInductance(w,betak,tl,Fl)

% ZHAT = DRT(BETAK,TL,FL) predicts impedance using the distribution of
% relaxation time model. 
% 
% input:
% (1) w: size: Jx1
%        class: double
%     definition: J is the number of data points
%     description: vector of angular frequencies
% (2) betak: size: Kx1
%            class: double
%     definition: K is the number of point parameters
%     description: vector of initial guesses for point parameter values
% (3) tl: size: Lx1
%         class: cell
%     definition: L is the number of processes
%     description: vector of distribution meshes; the l-th element of t1 is
%                  the mesh points of Fl
% (4) Fl: size: Lx1
%         class: cell
%     description: vector of distributions; the l-th element of Fl is the
%                  distribution of the l-th process
% 
% output:
% (1) Zhat: Zhat: size: Jx1
%           class: double
%     description: vector of predicted impedance values
% 
% Author: Surya Effendy
% Date: 05/14/2019

% We unpack the point parameters
Rinf=betak(1);
induc = betak(2);

% For convenience, we will pre-calculate w times exp(t1). Let's call this
% wtet1.
wtet1=bsxfun(@times,w,exp(tl{1}));

% Predict the impedance. We calculate the relaxation kernel:
K1=1./(1+1i*wtet1);

% Integrate using trapezoidal rule, then add Rinf and induc
Zhat=trapz(tl{1},bsxfun(@times,K1,Fl{1}),2)+Rinf + 1i*w*induc;

end