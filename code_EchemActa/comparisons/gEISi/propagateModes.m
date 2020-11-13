function g=propagateModes(g,modality,modalityNext,distType)

% G = PROPAGATEMODES(G,MODALITY,MODALITYNEXT,DISTTYPE) generates a new
% initial guess consisting of a sum of Dirac delta distributions based on
% the current maximum likelihood estimate. The algorithm works by matching
% the zeroth and first moments of the distributions, and approximately
% matching the higher moments.
% 
% input:
% (1) g: size: (K+3*(M1+M2+...))x1
%        class: double
%     definition: K is the number of point parameters; Ml is the number of
%                 basis functions needed to approximate the l-th
%                 distribution; (M1+M2+...) is the total number of basis
%                 functions used i.e. the complexity
%     description: vector of all parameter values
% (2) modality: size: Lx1
%               class: double
%     definition: L is the number of processes
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (3) modalityNext: size: Lx1
%                   class: double
%     description: vector of number of basis functions in the next-simplest
%                  distribution
% (4) distType: size: Lx1
%               class: cell
%     description: vector of the distributed nature of the processes; can
%                  be either 'series' or 'parallel'
% 
% output:
% (1) g: size: (K+3*(1+M1+M2+...))x1
%        class: double
%     description: vector of initial parameter values for the next-simplest
%                  distribution
% 
% Author: Surya Effendy
% Date: 03/10/2019

% Declare globals
global J K L w

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% Construct betak, Rmuwml, and we. These are easier to visualize.
betak=g(1:K);
Rmuwml=reshape(g(K+1:K+3*M),[M,3]);
we=g(K+3*M+1);
% Split Rmuwml to Rml, muml, and wml.
Rml=Rmuwml(:,1);
muml=Rmuwml(:,2);
wml=Rmuwml(:,3);

% The moment matching algorithm is immediately applicable for distributions
% whose nature is series. For distributions whose nature is parallel, we
% need the inverse of Rml:
Aml=1./Rml;

% Initialize empty sets for the parameters of the next-simplest
% distribution.
RmlNext=[];
mumlNext=[];
wmlNext=[];

% Iterate across all the distributions. If the number of basis functions
% intended to approximate the distribution has increased, perform moment
% matching. Otherwise, retain the parameter values.
% 
% To reiterate, the l-th element of modality is the number of basis
% functions needed to approximate the current maximum likelihood estimate
% of the l-th distribution, and the l-th element of modalityNext is the
% number of basis functions intended to approximate the l-th next-simplest
% distribution. 
sumModality=cat(1,0,cumsum(modality,1));

% Run through each distribution.
for i1=1:L
    % Case #1: No change in number of basis functions
    if modality(i1)==modalityNext(i1)
        RmlNext=cat(1,RmlNext,Rml(sumModality(i1)+1:sumModality(i1+1)));
        mumlNext=cat(1,mumlNext,muml(sumModality(i1)+1:sumModality(i1+1)));
        wmlNext=cat(1,wmlNext,wml(sumModality(i1)+1:sumModality(i1+1)));
    elseif strcmp(distType{i1},'series')
        % Case #2: The number of basis function has increased by 1, the
        % nature of the distribution is series.
        RmlTemp=cat(1,Rml(sumModality(i1)+1)/2,...
            Rml(sumModality(i1)+1:sumModality(i1+1)-1,:)/2+Rml(sumModality(i1)+2:sumModality(i1+1),:)/2,...
            Rml(sumModality(i1+1))/2);
        mumlTemp=cat(1,muml(sumModality(i1)+1)-exp(wml(sumModality(i1)+1)/2),...
            (Rml(sumModality(i1)+1:sumModality(i1+1)-1,:).*(muml(sumModality(i1)+1:sumModality(i1+1)-1,:)+exp(wml(sumModality(i1)+1:sumModality(i1+1)-1,:)/2))+...
            Rml(sumModality(i1)+2:sumModality(i1+1),:).*(muml(sumModality(i1)+2:sumModality(i1+1),:)-exp(wml(sumModality(i1)+2:sumModality(i1+1),:)/2)))./...
            (Rml(sumModality(i1)+1:sumModality(i1+1)-1,:)+Rml(sumModality(i1)+2:sumModality(i1+1),:)),...
            muml(sumModality(i1+1))+exp(wml(sumModality(i1+1))/2));
        wmlTemp=2*log(abs(log(w(J)/w(1))/(J-1)))*ones(modalityNext(i1),1);
        % Observe that some of the variables are indexed as Rml(...,:),
        % muml(...,:) and wml(...,:), and some are not. MATLAB recognizes
        % both row and column empty matrices. In this case, we want the
        % output to be such that, if the matrix is empty, then it is a
        % column empty matrix. This is done by adding the colon operator.
        
        % Pack these temporary variables into the permanent storage.
        RmlNext=cat(1,RmlNext,RmlTemp);
        mumlNext=cat(1,mumlNext,mumlTemp);
        wmlNext=cat(1,wmlNext,wmlTemp);
    elseif strcmp(distType{i1},'parallel')
        % Case #2: The number of basis function has increased by 1, the
        % nature of the distribution is parallel.
        AmlTemp=cat(1,Aml(sumModality(i1)+1)/2,...
            Aml(sumModality(i1)+1:sumModality(i1+1)-1,:)/2+Aml(sumModality(i1)+2:sumModality(i1+1),:)/2,...
            Aml(sumModality(i1+1))/2);
        mumlTemp=cat(1,muml(sumModality(i1)+1)-exp(wml(sumModality(i1)+1)/2),...
            (Aml(sumModality(i1)+1:sumModality(i1+1)-1,:).*(muml(sumModality(i1)+1:sumModality(i1+1)-1,:)+exp(wml(sumModality(i1)+1:sumModality(i1+1)-1,:)/2))+...
            Aml(sumModality(i1)+2:sumModality(i1+1),:).*(muml(sumModality(i1)+2:sumModality(i1+1),:)-exp(wml(sumModality(i1)+2:sumModality(i1+1),:)/2)))./...
            (Aml(sumModality(i1)+1:sumModality(i1+1)-1,:)+Aml(sumModality(i1)+2:sumModality(i1+1),:)),...
            muml(sumModality(i1+1))+exp(wml(sumModality(i1+1))/2));
        wmlTemp=2*log(abs(log(w(J)/w(1))/(J-1)))*ones(modalityNext(i1),1);
        
        % Pack these temporary variables into the permanent storage.
        RmlNext=cat(1,RmlNext,1./AmlTemp);
        mumlNext=cat(1,mumlNext,mumlTemp);
        wmlNext=cat(1,wmlNext,wmlTemp);
    end
end

% Pack everything together into the vector g.
g=cat(1,betak,RmlNext,mumlNext,wmlNext,we);

end