function [g,gn]=sample(fun,g,modality,annealMode,distType,dataType)

% [G,GN] = SAMPLE(FUN,G,MODALITY,ANNEALMODE,DISTTYPE,DATATYPE) estimates
% the maximum likelihood parameter values g and the Monte-Carlo Markov
% chain gn. The chain is only meaningful when simulated annealing is turned
% off. 
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
% (3) modality: size: Lx1
%               class: double
%     definition: L is the number of processes
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (4) annealMode: size: 1x2 or 1x3
%                 class: char
%     description: describes whether simulated annealing is active; can be
%                  either 'on' or 'off'
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
% (2) gn: size: (K+3*(M1+M2+...)+1)xN
%         class: double
%     definition: N is the number of Monte-Carlo Markov chain samples taken
%     description: matrix of sampled parameter values
% 
% Author: Surya Effendy
% Date: 03/09/2019

% Declare globals
global J K relTol

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% We start by estimating the appropriate step size:
[g,dg]=estimateStep(fun,g,[],modality,distType,dataType);

% Next, we estimate the number of successful Monte-Carlo steps needed to
% travel across the objective function surface. See the second manuscript.
if strcmp(dataType,'all')
    Nchain=2*(2*J)*round(norminv(1/2*(1+(1-relTol)^(1/(2*J))))^2/sqrt(K+3*M+1));
else
    Nchain=2*(J+1)*round(norminv(1/2*(1+(1-relTol)^(1/(J+1))))^2/sqrt(K+3*M+1));
end

% Perform the Monte-Carlo Markov chain sampling. Experience suggests that
% Gibbs sampling works better here. We generate a proposal step, and
% accept/reject the proposal. If accepted, we update the number of
% successful steps. If the proposed step is better than the current best,
% we re-estimate the step size. 
% 
% Initialize the number of successful steps and the number of elements in
% the Monte-Carlo Markov chain.
Nsuccess=1;
n=1;

% Initialize the temperature program. For simulated annealing, the
% temperature is inversely proportional to the number of successful steps.
if strcmp(annealMode,'on')
    T=@(Nsuccess)Nchain/Nsuccess;
else
    % Otherwise, the temperature is simply equal to 1, and we get back the
    % usual Monte-Carlo Markov chain formulation.
    T=@(Nsuccess)1;
end

% Create the sample chain.
gn=g;
% To be clear, gn is the sample chain and g is the estimate of the maximum
% likelihood parameters.

% Calculate the objective function value corresponding to the maximum
% likelihood parameters. This is also the current objective function value.
objFun=objectiveFunction(fun,g,modality,distType,dataType);
objFunNow=objFun;
% To be clear, objFun is the objective function at the maximum likelihood
% estimate, and objFunNow is the current objective function value. 

% Iterate the Monte-Carlo Markov chain sampling until the number of
% successful steps exceed Nchain.
while Nsuccess<Nchain
    % Perform Gibbs sampling. Gibbs sampling is Monte-Carlo Markov chain
    % done one parameter at a time. 
    for i1=1:(K+3*M+1)
        % Propose the next sample, perturbing the i-th parameter.
        gNext=gn(:,end);
        gNext(i1)=gn(i1,end)+randn*dg(i1);
        
        % Calculate the objective function value corresponding to the
        % proposed sample.
        objFunNext=objectiveFunction(fun,gNext,modality,distType,dataType);
        
        % Accept/reject the proposed sample. Note that we have appended all
        % samples, successful or otherwise, to the sample chain gn. This is
        % not right; in Gibbs sampling, we go through a cycle of proposals
        % through all the parameters, and then take only the last sample of
        % the cycle. The proposed sample must result in distributions with
        % positive mass.
        if rand<min(1,exp(-(objFunNext-objFunNow)/(2*T(Nsuccess)))*all(gNext(K+1:K+M)>=0)*all(diff(gNext(K+M+1:K+2*M))>=0))
            gn=cat(2,gn,gNext);
            objFunNow=objFunNext;
            
            % Update Nsuccess. Each successful Gibbs sampling counts as a
            % fraction of a successful Monte-Carlo Markov chain sample.
            Nsuccess=Nsuccess+1/(K+3*M+1);
        else
            gn=cat(2,gn,gn(:,end));
        end
        
        % Check if the current objective function value is lower than the
        % current best. If so, re-estimate the step size.
        if objFunNow<objFun            
            % Overwrite the maximum likelihood estimate
            g=gn(:,end);
            
            % Overwrite the step size, and update the maximum likelihood
            % estimate.
            [g,dg]=estimateStep(fun,g,dg,modality,distType,dataType);
            
            % Overwrite the corresponding objective function value.
            objFun=objectiveFunction(fun,g,modality,distType,dataType);
        end
    end
    
    % Erase the last (K+3*M+1) samples, and replace them with the last
    % sample. In effect, we have gone through a cycle of proposals through
    % all the parameters, and take only the last sample of the cycle.
    gNext=gn(:,n+K+3*M+1);
    gn(:,n+1:n+K+3*M+1)=[];
    gn=cat(2,gn,gNext);
    
    % Update the chain length.
    n=n+1;
end

end