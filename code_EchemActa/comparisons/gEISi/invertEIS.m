function [modality,betak,Rml,muml,wml,...
    betakn,Rmln,mumln,wmln,...
    wen,...
    tl,Fl]=invertEIS(fun,data,varargin)

% [MODALITY,BETAK,RML,MUML,WML,BETAKN,RMLN,MUMLN,WMLN,WEN,TL,FL] =
% INVERTEIS(FUN,DATA,DISTTYPE,BETAK,RTAUL,MUE) performs inversion proper on
% the EIS data. The algorithm is composed of two parts. The first discovers
% an estimate of the maximum likelihood estimate, as well as the associated
% complexity of the model. The second constructs the 95% credible interval
% of the model parameters and the distributions. Details on the first part
% can be found in the first part of the documentation. Likewise, details on
% the second part can be found in the second part of the documentation.
%
% input:
% (1) fun: size: 1x1
%          class: function_handle
%     description: calculates impedance given point and distributed
%                  parameters
% (2) data: size: Jx3
%           class: double
%     definition: J is the number of data points
%     description: 1st column containes the angular frequencies, 2nd column
%                  contains the real part of the impedance, and 3rd column
%                  contains the imaginary part of the impedance
% (3) distType: size: Lx1
%               class: cell
%     definition: L is the number of processes
%     description: vector of the distributed nature of the processes; can
%                  be either 'series' or 'parallel'
% (4) betak: size: Kx1
%            class: double
%     definition: K is the number of point parameters
%     description: vector of initial guesses for point parameter values
% (5) Rtaul: size: Lx2
%            class: double
%     description: matrix of initial guesses for the masses and the mean
%                  characteristic timescales of each process; 1st column is
%                  the mass and the 2nd column is the mean characteristic
%                  timescale
% (6) mue: size: 1x1
%          class: double
%     description: natural log of the square of the relative error of the
%                  measurement
%
% output:
% (1) modality: size: Lx1
%               class: double
%     description: vector of number of basis functions needed to estimate
%                  the l-th distribution
% (2) betak: size: Kx3
%            class: double
%     description: point parameter values; 1st column is the lower bound
%                  estimate, 2nd column is the maximum likelihood estimate,
%                  3rd column is the upper bound estimate
% (3) Rml: size: (M1+M2+...)x3
%          class: double
%     definition: Ml is the number of basis functions needed to approximate
%                 the l-th distribution; (M1+M2+...) is the total number of
%                 basis functions used i.e. the complexity
%     description: basis function masses; 1st column is the lower bound
%                  estimate, 2nd column is the maximum likelihood estimate,
%                  3rd column is the upper bound estimate
% (4) muml: size: (M1+M2+...)x3
%           class: double
%     description: basis function means; 1st column is the lower bound
%                  estimate, 2nd column is the maximum likelihood estimate,
%                  3rd column is the upper bound estimate
% (5) wml: size: (M1+M2+...)x3
%          class: double
%     description: basis function log-variances; 1st column is the lower
%                  bound estimate, 2nd column is the maximum likelihood
%                  estimate, 3rd column is the upper bound estimate
% (6) betakn: size: KxN
%             class: double
%     definition: N is the number of Monte-Carlo samples
%     description: matrix of point parameter samples
% (7) Rmln: size: (M1+M2+...)xN
%           class: double
%     description: matrix of basis function mass samples
% (8) mumln: size: (M1+M2+...)xN
%            class: double
%     description: matrix of basis function mean samples
% (9) wmln: size: (M1+M2+...)xN
%           class: double
%     description: matrix of basis function log-variance samples
% (10) wen: size: 1xN
%           class: double
%      description: vector of log-measurement variance samples
% (11) tl: size: Lx1
%          class: cell
%      description: vector of distribution meshes; the l-th element of t1
%                   is the mesh points of Fl
% (12) Fl: size: Lx1
%          class: cell
%      description: vector of distributions; the l-th element of Fl is the
%                   distribution of the l-th process
%
% Author: Surya Effendy
% Date: 03/08/2019

% Check amongst the standardized functions. For each standardized
% functions, return an initial guess.
functionHandle=functions(fun);
if strcmp(functionHandle.function,'DRT') && isempty(varargin)
    distType=cell(1,1);distType{1}='series';
    betak=min(data(:,2),[],1);
    Rtaul=[max(data(:,2),[],1)-min(data(:,2),[],1),NaN];
    theta=NaN;
elseif strcmp(functionHandle.function,'transmissiveDDT') && isempty(varargin)
    distType=cell(1,1);distType{1}='parallel';
    betak=min(data(:,2),[],1);
    Rtaul=[max(data(:,2),[],1)-min(data(:,2),[],1),NaN];
    theta=NaN;
else
    distType=varargin{1};
    betak=varargin{2};
    Rtaul=varargin{3};
    theta=varargin{4};
end

% Declare globals. Note that J is the number of data points, K is the
% number of point parameters, L is the number of processes, mue is the
% estimated mean of the log-measurement variance, sige is the standard
% deviation of the log-measurement variance, relTol is the relative
% tolerance for the convergence of the cross-validation problem, w is the
% vector of angular frequencies, and Z is the impedance. 
global J K L mue sige relTol w Z
J=size(data,1);
mue=theta;
sige=log(10)/2;

% Initialize K and L. The command 'size' is preferred over 'numel'.
K=size(betak,1);
L=size(Rtaul,1);

% Unpack data into the equivalent pair of w and Z
w=data(:,1);
Z=data(:,2)+1i*data(:,3);

% Parts of the initial guesses, i.e., betak, Rtaul, and mue may be missing.
% In which case, those missing parts are replaced with NaN. A crude guess
% for the initial guess is provided. There is no great guess for betak:
betak(isnan(betak))=1;
% For the first column of Rtaul, the initial guess can be set to the
% highest impedance, spread across all processes.
Rl=Rtaul(:,1);
Rl(isnan(Rl))=max(data(:,2),[],1)/L;
% For the second column of Rtaul, the initial guess can be set to the
% characteristic timescale of the experiment.
taul=Rtaul(:,2);
taul(isnan(taul))=1/sqrt(w(1)*w(J));
% Pack Rtaul
Rtaul=cat(2,Rl,taul);
% From experience, the log-measurement variance is approximately:
mue(isnan(mue))=-9;

% Initialize the modality. The term 'modality' is defined as the number of
% basis functions needed to approximate the l-th distribution corresponding
% to the l-th process. This also defined as Ml.
%
% In line with hypothesis making, we always start with the simplest
% possible approximation, in which each distribution is composed of a
% single basis function.
modality=enumerateModality([]);

% Declare the relative tolerance of the cross-validation procedure. The
% relative tolerance has a "natural" value of 0. However, for incomplete
% optimization routines, the relative tolerance should be increased. For
% the present work, a relative tolerance of 0.1. This tolerance corresponds
% to the construction of a 90% credible interval. In general, a more
% thorough optimization allows for a relative tolerance closer to 0.
relTol=0.1;

% We construct Rmuwml from Rtaul. Rmuwml is the compact matrix
% representation of the masses, means, and log-variances of all the basis
% functions used to represent all the distributions. As the name suggests,
% the 1st column is the mass, the 2nd column is the mean, and the 3rd
% column is the log-variance.
%
% With regards to the log-variance: we don't have a good guess for the
% log-variance, so we set it to be the "minimum", which is dictated by the
% experimental angular frequency scale.
w0=2*log(abs(log(w(J)/w(1)))/(J-1));
Rmuwml=[Rtaul(:,1),log(Rtaul(:,2)),w0*ones(L,1)];

% The log-measurement variance has been appended to the list of parameter
% values. As a initial guess, we assume that the simple model yields a
% reasonable estimate of the log-measurement variance. See crudeModel.m.
we=mue;

% Equivalently, the parameters can be expressed as the vector g containing
% all the parameter values. From top to bottom, gamma contains betak, Rml,
% muml, wml, and we.
g=cat(1,betak,Rmuwml(:),we);

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% Report current status of the program.
fprintf('Analyzing models of complexity %d\n',M);

% Set up subsequent optimization problems. Intuition suggests that the
% optimization scales with the square of the total number of parameters.
% Note that the total number of parameters is K+3*M+1; K point parameters,
% 3*M parameters for the basis functions, and the hyperparameterized
% log-measurement variance.
options=optimset('MaxFunEvals',100*(K+3*M+1)^2,'MaxIter',100*(K+3*M+1)^2);
% Setup the inequality constraint requiring the masses to be positive.
A1=cat(2,zeros(M,K),-eye(M),zeros(M,2*M+1));
A2=cat(2,zeros(M-1,K+M),cat(2,eye(M-1),zeros(M-1,1))-cat(2,zeros(M-1,1),eye(M-1)),zeros(M-1,M+1));
A=cat(1,A1,A2);
B=zeros(2*M-1,1);

% The optimization algorithm is run using all the data set. The
% optimization algorithm used is a combination of pattern search, simulated
% annealing, and diagonal search. The optimization algorithm can be
% alternatively used to get the credible interval. To activate simulated
% annealing, set annealMode to 'on'; to calculate credible interval, set
% annealMode to 'off'.
%
% The variable dataType describes the data used to perform the fitting.
% There are 3 possible dataTypes: 'all', in which the entire data set is
% used, 'real', in which part of the imaginary data set is discarded, and
% 'imag', im which part of the real data set is discarded.
dataType='all';
annealMode='on';
% Run simulated annealing.
[g,~]=sample(fun,g,modality,annealMode,distType,dataType);
% Run pattern search.
g=fmincon(@(g)objectiveFunction(fun,g,modality,distType,dataType),g,A,B,[],[],[],[],[],options);

% We now perform the real-imaginary cross-validation algorithm. We re-fit
% the parameter values using mostly the real part of the data set.
dataType='real';
annealMode='on';
% Initialize guess with g.
gR=g;
% Run simulated annealing.
[gR,~]=sample(fun,gR,modality,annealMode,distType,dataType);
% Run pattern search.
gR=fmincon(@(g)objectiveFunction(fun,g,modality,distType,dataType),gR,A,B,[],[],[],[],[],options);

% Calculate the imaginary part of the cross-validation error.
dataType='imag';
Xval=objectiveFunction(fun,gR,modality,distType,dataType);

% Next, we re-fit the parameter values using mostly the imaginary part of
% the data set:
dataType='imag';
annealMode='on';
% Initialize guess with g.
gI=g;
% Run simulated annealing.
[gI,~]=sample(fun,gI,modality,annealMode,distType,dataType);
% Run pattern search.
gI=fmincon(@(g)objectiveFunction(fun,g,modality,distType,dataType),gI,A,B,[],[],[],[],[],options);

% Calculate the real part of the cross-validation error.
dataType='real';
Xval=Xval+objectiveFunction(fun,gI,modality,distType,dataType);

% Store the result. Three things need to be stored: the previous maximum
% likelihood estimate (gPrev), the previous modality (modalityPrev), and
% the previous cross-validation error (XvalPrev).
gPrev=g;
modalityPrev=modality;
XvalPrev=Xval;

% We have completed the first iteration. We now move on to the second
% iteration. Generate the next-simplest distributions:
modalityNext=enumerateModality(modalityPrev);

% The matrix modalityNext is, in effect, a list of tasks to perform,
% wherein each task consists of performing real-imaginary cross-validation
% on a model with different number of basis functions per distribution.
for i1=1:L
    % Calculate the current complexity of the model, defined as the number of
    % parameters currently used to describe the underlying distribution of all
    % processes.
    M=sum(modalityNext(:,i1),1);
    
    % Report current status of the program.
    fprintf('Analyzing models of complexity %d, variant %d\n',M,i1);
    
    % Set up subsequent optimization problems.
    options=optimset('MaxFunEvals',100*(K+3*M+1)^2,'MaxIter',100*(K+3*M+1)^2);
    % Setup the inequality constraint requiring the masses to be positive.
    A1=cat(2,zeros(M,K),-eye(M),zeros(M,2*M+1));
    A2=cat(2,zeros(M-1,K+M),cat(2,eye(M-1),zeros(M-1,1))-cat(2,zeros(M-1,1),eye(M-1)),zeros(M-1,M+1));
    A=cat(1,A1,A2);
    B=zeros(2*M-1,1);
    
    % Set dataType and annealMode.
    dataType='all';
    annealMode='on';
    % Generate the initial guess using the current maximum likelihood
    % distributions. This is done by moment matching. We generate a sum of
    % Dirac delta distributions with zeroth and first moments equal to the
    % current maximum likelihood distributions.
    gNext=propagateModes(gPrev,modalityPrev,modalityNext(:,i1),distType);
    % Run simulated annealing.
    [gNext,~]=sample(fun,gNext,modalityNext(:,i1),annealMode,distType,dataType);
    % Run pattern search.
    gNext=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gNext,A,B,[],[],[],[],[],options);
    
    % We re-fit the parameter values using mostly the real part of the data
    % set.
    dataType='real';
    annealMode='on';
    % Initialize with gNext.
    gR=gNext;
    % Run simulated annealing.
    [gR,~]=sample(fun,gR,modalityNext(:,i1),annealMode,distType,dataType);
    % Run pattern search.
    gR=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gR,A,B,[],[],[],[],[],options);
    
    % Calculate the imaginary part of the cross-validation error.
    dataType='imag';
    XvalNext=objectiveFunction(fun,gR,modalityNext(:,i1),distType,dataType);
    
    % We re-fit the parameter values using mostly the imaginary part of the
    % data set.
    dataType='imag';
    annealMode='on';
    % Initialize with gNext.
    gI=gNext;
    % Run simulated annealing.
    [gI,~]=sample(fun,gI,modalityNext(:,i1),annealMode,distType,dataType);
    % Run pattern search.
    gI=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gI,A,B,[],[],[],[],[],options);
    
    % Calculate the real part of the cross-validation error.
    dataType='real';
    XvalNext=XvalNext+objectiveFunction(fun,gI,modalityNext(:,i1),distType,dataType);
    
    % Compare with the current best and overwrite.
    if XvalNext<Xval
        g=gNext;
        modality=modalityNext(:,i1);
        Xval=XvalNext;
    end
end

% Compare the current maximum likelihood estimate with the previous maximum
% likelihood estimate. While the cross-validation error decreases at a rate
% more rapid than the specified tolerance, repeat the algorithm.
while abs(XvalPrev-Xval)>relTol*4*J
    % Update the previous set of results.
    gPrev=g;
    modalityPrev=modality;
    XvalPrev=Xval;
    
    % Create the next list of tasks.
    modalityNext=enumerateModality(modalityPrev);
    
    % Run through the tasks.
    for i1=1:L
        % Calculate the current complexity of the model, defined as the number of
        % parameters currently used to describe the underlying distribution of all
        % processes.
        M=sum(modalityNext(:,i1),1);
        
        % Report current status of the program.
        fprintf('Analyzing models of complexity %d, variant %d\n',M,i1);
        
        % Set up subsequent optimization problems.
        options=optimset('MaxFunEvals',100*(K+3*M+1)^2,'MaxIter',100*(K+3*M+1)^2);
        % Setup the inequality constraint requiring the masses to be positive.
        A1=cat(2,zeros(M,K),-eye(M),zeros(M,2*M+1));
        A2=cat(2,zeros(M-1,K+M),cat(2,eye(M-1),zeros(M-1,1))-cat(2,zeros(M-1,1),eye(M-1)),zeros(M-1,M+1));
        A=cat(1,A1,A2);
        B=zeros(2*M-1,1);
        
        % Set dataType and annealMode.
        dataType='all';
        annealMode='on';
        % Generate the initial guess using the current maximum likelihood
        % distributions. This is done by moment matching. We generate a sum of
        % Dirac delta distributions with zeroth and first moments equal to the
        % current maximum likelihood distributions.
        gNext=propagateModes(gPrev,modalityPrev,modalityNext(:,i1),distType);
        % Run simulated annealing.
        [gNext,~]=sample(fun,gNext,modalityNext(:,i1),annealMode,distType,dataType);
        % Run pattern search.
        gNext=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gNext,A,B,[],[],[],[],[],options);
        
        % We re-fit the parameter values using mostly the real part of the data
        % set.
        dataType='real';
        annealMode='on';
        % Initialize with gNext.
        gR=gNext;
        % Run simulated annealing.
        [gR,~]=sample(fun,gR,modalityNext(:,i1),annealMode,distType,dataType);
        % Run pattern search.
        gR=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gR,A,B,[],[],[],[],[],options);
        
        % Calculate the imaginary part of the cross-validation error.
        dataType='imag';
        XvalNext=objectiveFunction(fun,gR,modalityNext(:,i1),distType,dataType);
        
        % We re-fit the parameter values using mostly the imaginary part of the
        % data set.
        dataType='imag';
        annealMode='on';
        % Initialize with gNext.
        gI=gNext;
        % Run simulated annealing.
        [gI,~]=sample(fun,gI,modalityNext(:,i1),annealMode,distType,dataType);
        % Run pattern search.
        gI=fmincon(@(g)objectiveFunction(fun,g,modalityNext(:,i1),distType,dataType),gI,A,B,[],[],[],[],[],options);
        
        % Calculate the real part of the cross-validation error.
        dataType='real';
        XvalNext=XvalNext+objectiveFunction(fun,gI,modalityNext(:,i1),distType,dataType);
        
        % Compare with the current best and overwrite.
        if XvalNext<Xval
            g=gNext;
            modality=modalityNext(:,i1);
            Xval=XvalNext;
        end
    end
end

% Step back. The current best cross-validation error does not differ from
% the previous by more than the specified tolerance. Restore the previous
% result.
g=gPrev;
modality=modalityPrev;

% The maximum likelihood estimate, as well as the optimal number of basis
% functions needed to approximate the distributions, have been obtained.
%
% Next, estimate the 95% credible interval. We generate multiple
% Monte-Carlo Markov chain samples using sample.m.
% 
% Set the number of multi-start iterations. In theory, one would suffice,
% but we will arbitrarily set this to 8.
Nms=8;

% Each multi-start iteration will generate a separate Monte-Carlo Markov
% chain, which will be compiled into a single matrix.
gn=[];

% Calculate the current complexity of the model, defined as the number of
% parameters currently used to describe the underlying distribution of all
% processes.
M=sum(modality,1);

% Set up subsequent optimization problems.
options=optimset('MaxFunEvals',100*(K+3*M+1)^2,'MaxIter',100*(K+3*M+1)^2);
% Setup the inequality constraint requiring the masses to be positive.
A1=cat(2,zeros(M,K),-eye(M),zeros(M,2*M+1));
A2=cat(2,zeros(M-1,K+M),cat(2,eye(M-1),zeros(M-1,1))-cat(2,zeros(M-1,1),eye(M-1)),zeros(M-1,M+1));
A=cat(1,A1,A2);
B=zeros(2*M-1,1);

% Set dataType.
dataType='all';

% Run pattern search.
g=fmincon(@(g)objectiveFunction(fun,g,modality,distType,dataType),g,A,B,[],[],[],[],[],options);
for i1=1:Nms
    % Monte-Carlo samples require a burn-in, wherein the initial part of
    % the chain is discarded, so as to decorrelate the samples from the
    % initial set of parameter values.
    % 
    % Burn-in is done using simulated annealing. Notice that the sequence
    % burn-in / sample is repeated Nms times. This returns Nms ideally
    % independent Monte-Carlo Markov chain samples.
    annealMode='on';
    [gBurn,gnBurn]=sample(fun,g,modality,annealMode,distType,dataType);
    
    % Obtain the Monte-Carlo samples proper.
    annealMode='off';
    [gTemp,gnTemp]=sample(fun,gnBurn(:,end),modality,annealMode,distType,dataType);
    
    % Compare gBurn and gTemp, and take the better result
    if objectiveFunction(fun,gBurn,modality,distType,dataType)<objectiveFunction(fun,gTemp,modality,distType,dataType)
        g=gBurn;
    else
        g=gTemp;
    end
    
    % Append the resulting Monte-Carlo samples to gn.
    gn=cat(2,gn,gnTemp);
    
    % Report current status of the program.
    fprintf('Calculating credible interval: set %d out of %d\n',i1,Nms);
end

% Unpack g and gn into betak, Rml, muml, wml, betakn, Rmln, mumln,
% wmln, and wen.
betak=g(1:K);
betakn=gn(1:K,:);
Rml=g(K+1:K+M);
Rmln=gn(K+1:K+M,:);
muml=g(K+M+1:K+2*M);
mumln=gn(K+M+1:K+2*M,:);
wml=g(K+2*M+1:K+3*M);
wmln=gn(K+2*M+1:K+3*M,:);
wen=gn(K+3*M+1,:);

% Get the 95% credible interval. We do this by sorting each sample chain in
% ascending order, and then discarding the top and bottom 2.5%.
N=size(gn,2);
Ntrim=round(N/40);

% Discard betak samples.
betaknTemp=sort(betakn,2,'ascend');
betaknTemp(:,N-Ntrim+1:N)=[];
betaknTemp(:,1:Ntrim)=[];

RmlnTemp=sort(Rmln,2,'ascend');
RmlnTemp(:,N-Ntrim+1:N)=[];
RmlnTemp(:,1:Ntrim)=[];

mumlnTemp=sort(mumln,2,'ascend');
mumlnTemp(:,N-Ntrim+1:N)=[];
mumlnTemp(:,1:Ntrim)=[];

wmlnTemp=sort(wmln,2,'ascend');
wmlnTemp(:,N-Ntrim+1:N)=[];
wmlnTemp(:,1:Ntrim)=[];

% Append the credible interval.
betak=cat(2,betaknTemp(:,1),betak,betaknTemp(:,end));
Rml=cat(2,RmlnTemp(:,1),Rml,RmlnTemp(:,end));
muml=cat(2,mumlnTemp(:,1),muml,mumlnTemp(:,end));
wml=cat(2,wmlnTemp(:,1),wml,wmlnTemp(:,end));

% Finally, construct the 95% credible interval for the probability
% distributions Fl.
%
% Initialize tl and Fl.
tl=cell(L,1);
Fl=cell(L,1);

% For each distribution, generate a mesh. Then calculate a distribution
% correspoding to each sample in gn.
for i1=1:L
    % Unpack the distributed parameters. Use the vector modality as a guide
    % for breaking up the parameters corresponding to different processes.
    sumModality=cat(1,0,cumsum(modality));
    RmlTemp=Rml(sumModality(i1)+1:sumModality(i1+1),2);
    mumlTemp=muml(sumModality(i1)+1:sumModality(i1+1),2);
    wmlTemp=wml(sumModality(i1)+1:sumModality(i1+1),2);
    
    % Generate the mesh.
    tl{i1}=generateMesh(mumlTemp,wmlTemp);
    Ol=size(tl{i1},2);
    
    % Evaluate the maximum likelihood distribution
    [~,FlTemp]=evaluateDistribution(RmlTemp,mumlTemp,wmlTemp,distType{i1});
    
    % Generate a placeholder for the distributions corresponding to gn.
    FlnTemp=zeros(N,Ol);
    
    % Run through each sample.
    for i2=1:N
        % Unpack the distributed parameters. Use the vector modality as a guide
        % for breaking up the parameters corresponding to different processes.
        sumModality=cat(1,0,cumsum(modality));
        RmlTemp=Rmln(sumModality(i1)+1:sumModality(i1+1),i2);
        mumlTemp=mumln(sumModality(i1)+1:sumModality(i1+1),i2);
        wmlTemp=wmln(sumModality(i1)+1:sumModality(i1+1),i2);
        
        [~,FlnTemp(i2,:)]=evaluateDistribution(RmlTemp,mumlTemp,wmlTemp,distType{i1},tl{i1});
    end
    
    % Sort the distributions in ascending order, and discard the top and
    % bottom 2.5% to get the 95% credible interval.
    FlnTemp=sort(FlnTemp,1,'ascend');
    FlnTemp(N-Ntrim+1:N,:)=[];
    FlnTemp(1:Ntrim,:)=[];
    
    % Append the credible interval
    Fl{i1}=cat(1,FlnTemp(1,:),FlTemp,FlnTemp(end,:));
end

end