function model = sms_DRT(Z_real_in, Z_imag_in, A_real, A_imag, Lambda, i_L, f)
% estimating drt given impedance measurements, basis matrix, 
% and search grid of shirnkage paparemeter

% Inputs:
% Z_real_in: real-part of impedance measurments
% Z_imag_in: imaginary-part of impedance measuremtns
% A_real: real part of basis matrix, via cal_Basis function
% A_imag: imaginary part of basis matrix, via cal_Basis function
% lambda: vector containing multiple shrinkage tuning parameter values
% i_L: 0 or 1, default is 0. 0: no high-frequency distortion inductance, 1: with high-frequency inducance
% f: frquency sampling points of impedance data, only needed when i_L=1.

% Output:
% out.R_infy: estimated high-frequency cut-off resistance
% out.R_p: estimated polarization resistance
% out.beta: estimated DRT
% out.Z_real: real-part of fitted impedance
% out.Z_imag: imaginary-part of fitted impedance
% out.inductance: estimated high-frequency distortion inductance

%Reference: https://doi.org/10.1016/j.electacta.2019.05.010

if nargin < 6
  i_L = 0;
end

nl = length(Lambda);
[nf,nt] = size(A_imag);
BETA = zeros(nt,nl);
RP = zeros(1,nl);
R_INFY = zeros(1,nl);
L = zeros(1,nl);
Error = zeros(1,nl);

for k = 1:nl
    lambda = Lambda(k);
    if i_L == 0
        inner_model = NN_LARS(A_imag,Z_imag_in,lambda);
        [~,index] = min(inner_model.Cp);
        beta_aug = inner_model.X(:,index);
        beta_hat = (1+lambda).^(0.5).*beta_aug;
        rp = sum(beta_hat);
    else
        l = 0;
        for m =1:20
            inner_model = NN_LARS(A_imag,Z_imag_in-2*pi*l.*f,lambda);
            [~,index] = min(inner_model.Cp);
            beta_aug = inner_model.X(:,index);
            beta_hat = (1+lambda).^(0.5).*beta_aug;
            rp = sum(beta_hat);
            l = lsqnonneg(2*pi.*f,Z_imag_in-(A_imag*beta_hat));
        end
        L(k) = l;
    end
   
    r_infy = sum(Z_real_in-(A_real*beta_hat))./nf;
    
    BETA(:,k) = beta_hat;
    RP(k) = rp;
    R_INFY(k) = r_infy;
    re_err = (Z_real_in - r_infy - A_real*beta_hat)'*(Z_real_in - r_infy - A_real*beta_hat);
    if i_L == 0
        im_err = (Z_imag_in - A_imag*beta_hat)'*(Z_imag_in - A_imag*beta_hat);
    else
        im_err = (Z_imag_in - 2*pi*l.*f-A_imag*beta_hat)'*(Z_imag_in - 2*pi*l.*f-A_imag*beta_hat);
    end
    Error(k) = re_err + im_err;
end

[~,IND] = min(Error);

model.R_p = RP(IND);
model.R_infy = R_INFY(IND);
model.beta = BETA(:,IND)./RP(IND);
model.Z_real = R_INFY(IND) + A_real*BETA(:,IND);
if i_L == 0
    model.inductance = 'not considered';
    model.Z_imag = A_imag*BETA(:,IND);
else
    model.inductance = L(IND);
    model.Z_imag = A_imag*BETA(:,IND) + 2*pi*L(IND).*f;
end
end

function [out] = NN_LARS(A, b, delta)
% Non-negative LARS algorithm
% based on code of Martin Slawski, paper DOI: 10.1214/13-EJS868
  
%%% Initialize objective -- global [possibly unused]
global f;
f = @(x) norm(A * x - b)^2;

[n p] = size(A);

A = sqrt(1+delta).*[A;eye(p)];
b = [b;zeros(p,1)]; 

[n p] = size(A);

b0 = pinv(A)*b;
sigma2e = sum((b - A*b0).^2)/n; % Mean Square Error of low-bias model

indices = (1:p)';
%initialize variables
nvars = min(n, p);
maxk = Inf; % Maximum number of iterations
maxvar = 0; % early stopping
I = 1:p; % inactive set

P = []; % positive set
% we shall always use the Cholesky factorization
R = []; % Cholesky factorization R'R = X'X where R is upper triangular


lassocond = 0; % LASSO condition boolean
earlystopcond = 0; % Early stopping condition boolean
k = 0; % Iteration count
vars = 0; % Current number of variables
x = zeros(p, 1);

c = A' * b;

out.err = [];
out.X =[];
out.df = [];
out.Cp = [];

%%% LARS main loop
while vars < nvars && ~earlystopcond && k < maxk
  k = k + 1;
  [C j] = max(c(I));
  if C < eps
     AS = indices(x ~= 0); % active set
     out.df(k) = length(AS);
     out.err(k) = f(x);
     out.X(:,k) = x;
     out.Cp(k) = out.err(k)/sigma2e - n + 2*out.df(k);
     return; 
  end
  j = I(j); % add one variable at a time only.

  if ~lassocond % if a variable has been dropped, do one iteration with this configuration (don't add new one right away)
    R = cholinsert(R, A(:,j), A(:,P));
    P = [P j];
    I(I == j) = [];
    vars = vars + 1;
  end

    s = ones(vars, 1);

    GA1 = R\(R'\s);
    AA = 1/sqrt(sum(GA1));
    w = AA*GA1;

    u = A(:,P)*w; % equiangular direction (unit vector)
  if vars == nvars % if all variables active, go all the way to the lsq solution
    gamma = C/AA;
  else
    a = A'*u; % correlation between each variable and equiangular vector
    temp = (C - c(I))./(AA - a(I)); % note: only positive correlations matter.
    gamma = min([temp(temp > eps); C/AA]);
  end

  % LASSO modification

    lassocond = 0;
    temp = (-x(P)./w)';
    %findtemp = find(temp > 0);
    [gamma_tilde] = min([temp(temp > 0) gamma]);
    j = find(abs(temp - gamma_tilde) < eps);
    if gamma_tilde < gamma
      gamma = gamma_tilde;
      lassocond = 1;
    end

  x(P) = x(P) + gamma*w;

  % If LASSO condition satisfied, drop variable from active set
  if lassocond == 1
      lj = length(j);
      for jj = 1:lj
        R = choldown(R, j(lj - jj + 1));
        I = [I P(j)];
      end
      P(j) = [];
      vars = vars - length(j); % note that in general one may have several drops at a time.
  end

  % Early stopping at specified number of variables
  if maxvar > 0
    earlystopcond = vars >= maxvar;
  end

  c = A'*(b - A(:,P) * x(P));

  AS = indices(x ~= 0); % active set
  out.df(k) = length(AS);
  out.err(k) = f(x);
  out.X(:,k) = x;
  out.Cp(k) = out.err(k)/sigma2e - n + 2*out.df(k);

  if not(all(isfinite(x)))
      break;
  end
end

if k == maxk
  disp('LARS warning: Forced exit. Maximum number of iteration reached.');
end
end

function R = cholinsert(R, x, X)
diag_k = x'*x; % diagonal element k in X'X matrix
if isempty(R)
  R = sqrt(diag_k);
else
  col_k = x'*X; % elements of column k in X'X matrix
  R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
  R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
  R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
end
end

function Lkt =  choldown(Lt, k) 

p = length(Lt);

% drop the kth clm of Lt
Temp = Lt;
Temp(:,k) = []; % Temp in R^(p,p-1)

% Givens Rotations
for i = k:p-1,
    a = Temp(i,i);
    b = Temp(i+1,i);
    r = sqrt(sum(Lt(:,i+1).^2) - sum(Temp(1:i-1,i).^2));
    c =  r * a / (a^2+b^2);
    s =  r * b / (a^2+b^2);
    % ith row of rotation matrix H
    Hrowi = zeros(1,p); Hrowi(i) = c; Hrowi(i+1) = s; 
    % (i+1)th row of ration matrix H
    Hrowi1 = zeros(1,p); Hrowi1(i) = -s; Hrowi1(i+1) = c;
    % modify the ith and (i+1)th rows of Temp
    v = zeros(2,p-1);
    v(1,i:p-1) = Hrowi * Temp(:,i:p-1);
    v(2,i+1:p-1) = Hrowi1 * Temp(:,i+1:p-1);
    Temp(i:i+1,:) =  v;
end

% drop the last row
Lkt = Temp(1:p-1,:);
end