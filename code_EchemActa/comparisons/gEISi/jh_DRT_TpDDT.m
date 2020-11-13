function Zhat = jh_DRT_TpDDT(w,betak,tl,Fl)
% DRT and transmissive planar DDT; DDT at higher timescale

% We unpack the point parameters. For this sample problem, there is only
% one point parameter, which is Rinf.
Rinf=betak;

% For convenience, we will pre-calculate w times exp(t1). Let's call this
% wtet1.
wtet1=bsxfun(@times,w,exp(tl{1}));
wtet2=bsxfun(@times,w,exp(tl{2}));


% DRT kernel
K1=1./(1+1i*wtet1);
% Integrate using trapezoidal rule
Z1=trapz(tl{1},bsxfun(@times,K1,Fl{1}),2);

% Tp-DDT kernel
K2 = sqrt(1i*wtet2)./tanh(sqrt(1i*wtet2));
% The calculation of the diffusion kernel involves division between two
% small numbers. We correct this manually.
K2(isnan(K2))=1;
% Integrate using trapezoidal rule
Y2=trapz(tl{2},bsxfun(@times,K2,Fl{2}),2);

% sum impedance contributions
Zhat = Z1 + 1./Y2 + Rinf;





    