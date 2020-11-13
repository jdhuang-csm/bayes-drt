function [g,dg]=estimateStep(fun,g,dg,modality,distType,dataType)

% [G,DG] = ESTIMATESTEP(FUN,G,DG,MODALITY,DISTTYPE,DATATYPE) performs a
% diagonal search followed by a quadratic estimation of the objective
% function surface. The estimated surface is then used to calculate an
% appropriate step size for the Monte-Carlo Markov chain.
%
% input:
% (1) fun: size: 1x1
%          class: function_handle
%     description: calculates impedance given point and distributed
%                  parameters
% (2) g: size: (K+3*(M1+M2+...))x1
%        class: double
%     definition: K is the number of point parameters; Ml is the number of
%                 basis functions needed to approximate the l-th
%                 distribution; (M1+M2+...) is the total number of basis
%                 functions used i.e. the complexity
%     description: vector of all parameter values
% (3) dg: size: (K+3*(M1+M2+...)+1)x1
%         class: double
%     description: vector of Monte-Carlo Markov chain step sizes
% (4) modality: : size: Lx1
%               class: double
%     definition: L is the number of processes
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (5) distType: size: Lx1
%               class: cell
%     description: vector of the distributed nature of the processes; can
%                  be either 'series' or 'parallel'
% (6) dataType: size: 1x3 or 1x4
%               class: char
%     description: describes the data used to perform maximum likelihood
%                  estimation; can be 'all', 'real', or 'imag'
%
% output:
% (1) g: size: (K+3*(M1+M2+...)+1)x1
%        class: double
%     description: vector of estimated maximum likelihood parameter values
% (2) dg: size: (K+3*(M1+M2+...)+1)x1
%         class: double
%     description: vector of Monte-Carlo Markov chain step sizes
%
% Author: Surya Effendy
% Date: 03/09/2019

% Declare globals
global J K sige w

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% Perform the diagonal search. Diagonal search is done by optimizing a few
% parameters at a time.
% parameter at a time. We start with the log-variances, since they are the
% most poorly estimated, then the masses, the means, the point parameters,
% and the measurement error.
searchSet=cat(2,K+2*M+1:K+3*M,K+1:K+2*M,1:K,K+3*M+1);
for i1=searchSet
    % Optimize, constraining the mass to be positive and the timescales to
    % be appropriately ordered
    if i1>=K+1 && i1<=K+M
        % Construct an anonymous function for the i1-th parameter
        altFun=@(altg)objectiveFunction(fun,cat(1,g(1:i1-1),altg,g(i1+1:K+3*M+1)),modality,distType,dataType);
        g(i1)=fmincon(altFun,g(i1),[],[],[],[],0,[]);
    elseif i1==K+M+1
        altFun=@(altg)objectiveFunction(fun,cat(1,g(1:i1-1),altg,g(i1+M:K+3*M+1)),modality,distType,dataType);
        A=cat(2,eye(M-1),zeros(M-1,1))-cat(2,zeros(M-1,1),eye(M-1));
        B=zeros(M-1,1);
        g(i1:i1+M-1)=fmincon(altFun,g(i1:i1+M-1),A,B,[],[],[],[]);
    elseif i1<=K || i1>+K+2*M+1
        % Construct an anonymous function for the i1-th parameter
        altFun=@(altg)objectiveFunction(fun,cat(1,g(1:i1-1),altg,g(i1+1:K+3*M+1)),modality,distType,dataType);
        g(i1)=fminsearch(altFun,g(i1));
    end
end

% Initialize dg. We want to start by constructing a "default" estimate for
% dg:
betak=g(1:K);
Rmuwml=reshape(g(K+1:K+3*M),[M,3]);
% Split Rmuwml to Rml, muml, and wml.
Rml=Rmuwml(:,1);

% The estimates are based on the natural scales of the parameter
% values.
dbetak=betak/J;
dRml=Rml/J;
dmuml=2*abs(log(w(J)/w(1)))/(J-1)*ones(M,1);
% The natural scale of wml can be shown to be inversely dependent on
% exp(wml/2):
%     dwml=2*abs(log(w(J)/w(1)))/(J-1)./exp(wml/2)
% This is a problem, because for the Dirac delta distribution, the
% natural scale diverges. We replace it with the crude natural scale:
dwml=2*abs(log(w(J)/w(1)))/(J-1)*ones(M,1);
dwe=sige;

% Pack the estimates
dg0=cat(1,dbetak,dRml,dmuml,dwml,dwe);

% If we do not have an estimate for dg, insert the default
if isempty(dg)
    dg=dg0;
end

% Otherwise dg already exists. We move on to obtaining a refined estimate.
% We calculate the objective function once at the base.
objFun=objectiveFunction(fun,g,modality,distType,dataType);

% Calculate the proper perturbation. The proper perturbation is the change
% in the value of the objective function which results in a moderate
% Monte-Carlo acceptance probability. See the second manuscript.
dObjFun=2;
for i1=1:(K+3*M+1)
    % The refined estimate is obtained by estimating a quadratic surface
    % about the objective function. We do this one parameter at a time.
    gPlus=g;
    gPlus(i1)=g(i1)+dg(i1);
    objFunPlus=objectiveFunction(fun,gPlus,modality,distType,dataType);
    
    gMinus=g;
    gMinus(i1)=g(i1)-dg(i1);
    objFunMinus=objectiveFunction(fun,gMinus,modality,distType,dataType);
    
    % Calculate the coefficients of the quadratic approximation to the
    % objective function surface with respect to one of the parameters.
    v=[1,g(i1)-dg(i1),(g(i1)-dg(i1))^2;...
        1,g(i1),g(i1)^2;...
        1,g(i1)+dg(i1),(g(i1)+dg(i1))^2]\[objFunMinus;objFun;objFunPlus];
    c=v(1);b=v(2);a=v(3);
    
    % Consider all possible shapes of the quadratic approximation, and
    % choose a step size based on the proper perturbation.
    gPlusFun=@(objFunTarget)(-b+sqrt(b^2-4*a*(c-objFunTarget)))/(2*a);
    gMinusFun=@(objFunTarget)(-b-sqrt(b^2-4*a*(c-objFunTarget)))/(2*a);
    
    % Case #1
    if objFun<objFunMinus && objFunMinus<objFunPlus
        objFunTarget=objFun+dObjFun;
        dg(i1)=gPlusFun(objFunTarget)-g(i1);
    elseif objFun<objFunPlus && objFunPlus<objFunMinus
        objFunTarget=objFun+dObjFun;
        dg(i1)=g(i1)-gMinusFun(objFunTarget);
    elseif objFun>objFunMinus && objFunMinus>objFunPlus
        objFunTarget=objFun-dObjFun;
        dg(i1)=gPlusFun(objFunTarget)-g(i1);
    elseif objFun>objFunPlus && objFunPlus>objFunMinus
        objFunTarget=objFun-dObjFun;
        dg(i1)=g(i1)-gMinusFun(objFunTarget);
    elseif objFunMinus<objFun && objFun<objFunPlus && a>=0
        objFunTarget=objFun+dObjFun;
        dg(i1)=gPlusFun(objFunTarget)-g(i1);
    elseif objFunMinus>objFun && objFun>objFunPlus && a>=0
        objFunTarget=objFun+dObjFun;
        dg(i1)=g(i1)-gMinusFun(objFunTarget);
    elseif objFunMinus<objFun && objFun<objFunPlus && a<0
        objFunTarget=objFun-dObjFun;
        dg(i1)=g(i1)-gMinusFun(objFunTarget);
    elseif objFunMinus>objFun && objFun>objFunPlus && a<0
        objFunTarget=objFun-dObjFun;
        dg(i1)=gPlusFun(objFunTarget)-g(i1);
    end
end

% Several "off" cases may occur. For very flat functions, dg will be large.
% This is the case for, for example, the log-variance of the Dirac delta
% distribution. We need to bound it. This is done by setting an upper bound
% equal to the natural scales:
dg=min(abs(dg),abs(dg0));

end