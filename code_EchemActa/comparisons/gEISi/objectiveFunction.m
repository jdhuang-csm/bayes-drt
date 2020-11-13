function objFun=objectiveFunction(fun,g,modality,distType,dataType)

% OBJFUN = OBJECTIVEFUNCTION(FUN,G,MODALITY,DISTTYPE,DATATYPE)
% calculates the objective function needed to solve the internal and
% external optimization problems. 
% 
% input:
% (1) fun: size: 1x1
%          class: function_handle
%     description: calculates impedance given point and distributed
%                  parameters
% (2) g: size: (K+3*(M1+M2+...)+1)x1
%        class: double
%     definition: K is the number of point parameters; Ml is the number of
%                 basis functions needed to approximate the l-th
%                 distribution; (M1+M2+...) is the total number of basis
%                 functions used i.e. the complexity
%     description: vector of all parameter values
% (3) modality: size: Lx1
%               class: double
%     definition: L is the number of processes
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (4) distType: size: Lx1
%               class: cell
%     description: vector of the distributed nature of the processes; can
%                  be either 'series' or 'parallel'
% (5) dataType: size: 1x3 or 1x4
%               class: char
%     description: describes the data used to perform maximum likelihood
%                  estimation; can be 'all', 'real', or 'imag'
% 
% outer:
% (1) objFun: size: 1x1
%             class: double
%     description: objective function value
% 
% Author: Surya Effendy
% Date: 03/09/2019

% Declare globals 
global J K mue sige Z

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% Unpack the vector back into betak and Rmuwml. Notice that we are
% continuously switching between g and (betak and Rmuml). This is because
% the former is required by MATLAB syntax, but the latter is easier to
% visualize.
betak=g(1:K);
Rmuwml=reshape(g(K+1:K+3*M),[M,3]);
we=g(K+3*M+1);
% Split Rmuwml to Rml, muml, and wml.
Rml=Rmuwml(:,1);
muml=Rmuwml(:,2);
wml=Rmuwml(:,3);

% Get model prediction of impedance
Zhat=exactModel(fun,betak,Rml,muml,wml,modality,distType);

% Calculate the objective function and the chi2 values depending on the
% dataType under consideration. For dataType = 'all', the chi2 is
% constructed using the normalized real and imaginary parts of the data
% set. For dataType = 'real', the chi2 is constructed using the normalized
% real part of the data set, as well as the mean of the normalized
% imaginary part of the data set. Ditto for dataType = 'imag'.
% 
% Notice that the form of the error is assumed to be 
%     Zhat = Z + delta*abs(Z)*(N(0,1)+1i*N(0,1))
% There is a weak reasoning behind this. High-frequency measurements tend
% to contain more replicates than low-frequency measurements. They also
% tend to be smaller in magnitude. Thus, we assume that the resulting
% "averaged" measurement error is small at low frequencies.
if strcmp(dataType,'all')
    chi2=sum((real(Z-Zhat).^2+imag(Z-Zhat).^2)./abs(Z).^2/exp(we),1);
    objFun=chi2+(mue-we)^2/sige^2+we*2*J;
elseif strcmp(dataType,'real')
    chi2=sum(real(Z-Zhat).^2./abs(Z).^2/exp(we),1)+...
        1/J*(sum(imag(Z-Zhat)./abs(Z),1)^2)/exp(we);
    objFun=chi2+(mue-we)^2/sige^2+we*(J+1);
elseif strcmp(dataType,'imag')
    chi2=sum(imag(Z-Zhat).^2./abs(Z).^2/exp(we),1)+...
        1/J*(sum(real(Z-Zhat)./abs(Z),1)^2)/exp(we);
    objFun=chi2+(mue-we)^2/sige^2+we*(J+1);
end

end