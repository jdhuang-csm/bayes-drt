function Zhat = jh_DRT_TpDDT_BpDDT(w,betak,tl,Fl)
% DRT and transmissive planar DDT and blocking planar DDT
% tau_DRT < tau_TpDDT < tau_BpDDT

% We unpack the point parameters. For this sample problem, there is only
% one point parameter, which is Rinf.
Rinf=betak;

% For convenience, we will pre-calculate w times exp(t1). Let's call this
% wtet1.
wtet1=bsxfun(@times,w,exp(tl{1}));
wtet2=bsxfun(@times,w,exp(tl{2}));
wtet3=bsxfun(@times,w,exp(tl{3}));

% DRT kernel
K1=1./(1+1i*wtet1);
% Integrate using trapezoidal rule
Z1=trapz(tl{1},bsxfun(@times,K1,Fl{1}),2);

% TP-DDT kernel
K2 = sqrt(1i*wtet2)./tanh(sqrt(1i*wtet2));
% The calculation of the diffusion kernel involves division between two
% small numbers. We correct this manually.
K2(isnan(K2))=1;
% Integrate using trapezoidal rule
Y2=trapz(tl{2},bsxfun(@times,K2,Fl{2}),2);

% BP-DDT kernel
K3 = sqrt(1i*wtet3)./coth(sqrt(1i*wtet3));
% The calculation of the diffusion kernel involves division between two
% small numbers. We correct this manually.
K3(isnan(K3))=1;
% Integrate using trapezoidal rule
Y3=trapz(tl{3},bsxfun(@times,K3,Fl{3}),2);

% sum impedance contributions
Zhat = Z1 + 1./Y2 + 1./Y3 + Rinf;





    