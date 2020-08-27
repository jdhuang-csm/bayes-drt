function [A_real, A_imag] = cal_Basis(f,t,K)
%Calculate basis matrix A_real, A_imag, given frequency f and time f vector
% f: frequency sampling point of impedance data
% t: time sampling point of drt domain
% K: random sampling size, default: K = 1e6

% Reference: https://doi.org/10.1016/j.electacta.2019.05.010
if nargin < 3
  K = 1e6;
end

nf = length(f);
nt = length(t);
A_real = zeros(nf,nt-1);
A_imag = zeros(nf,nt-1);

for i = 1:nf
    for j = 1:nt-1
        sv = unifrnd(t(j),t(j+1),K,1);
        cpx = sum(1./(1+2i*pi*f(i).*sv))/K;
        
        A_real(i,j) = real(cpx);
        A_imag(i,j) = imag(cpx);
    end
end
end