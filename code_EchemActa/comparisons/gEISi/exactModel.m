function Zhat=exactModel(fun,betak,Rml,muml,wml,modality,distType)

% ZHAT = EXACTMODEL(FUN,BETAK,RML,MUML,WML,MODALITY,DISTTYPE) predicts
% impedance using the distributed model, wherein the impedance is assumed
% to arise due to some distribution in parameter values, either in parallel
% or in series. 
% 
% input:
% (1) fun: size: 1x1
%          class: function_handle
%     description: calculates impedance given point and distributed
%                  parameters
% (2) betak: size: Kx1
%            class: double
%     definition: K is the number of point parameters
%     description: vector of initial guesses for point parameter values
% (3) Rml: size: (M1+M2+...)x1
%          class: double
%     definition: Ml is the number of basis functions needed to approximate
%                 the l-th distribution; (M1+M2+...) is the total number of
%                 basis functions used i.e. the complexity
%     description: vector of basis function masses
% (4) muml: size: (M1+M2+...)x1
%           class: double
%     description: vector of basis function means
% (5) wml: size: (M1+M2+...)x1
%          class: double
%     description: vector of basis function log-variances
% (6) modality: size: Lx1
%               class: double
%     definition: L is the number of processes
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (7) distType: size: Lx1
%               class: cell
%     description: vector of the distributed nature of the processes; can
%                  be either 'series' or 'parallel'
% 
% output:
% (1) Zhat: Zhat: size: Jx1
%           class: double
%     description: vector of predicted impedance values
% 
% Author: Surya Effendy
% Date: 03/12/2019

% Declare globals.
global w L

% Initial tl and Fl. These are the distributions corresponding to each
% process.
tl=cell(L,1);
Fl=cell(L,1);

% We unpack the distributed parameters. Here we use the vector modality as
% a guide for breaking up parameters corresponding to different processes.
sumModality=cat(1,0,cumsum(modality));
for i1=1:L
    RTemp=Rml(sumModality(i1)+1:sumModality(i1+1));
    muTemp=muml(sumModality(i1)+1:sumModality(i1+1));
    wTemp=wml(sumModality(i1)+1:sumModality(i1+1));
    
    % Evaluate the distribution for each process. 
    [tl{i1},Fl{i1}]=evaluateDistribution(RTemp,muTemp,wTemp,distType{i1});
end

% Apply the user-defined function to calculate impedance.
Zhat=fun(w,betak,tl,Fl);

end