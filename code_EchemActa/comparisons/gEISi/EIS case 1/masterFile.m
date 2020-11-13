% Generalized EIS (gEIS) inversion code returns point and distributed
% parameter estimates, as well as the 90% credible interval of those
% estimates, given model and data. For EIS case 1 through 7, the input
% model is named myFun.m.
% 
% On naming convention. For all variables, the index goes from 1 to the
% capital letter of the index. There are 6 indexes. 'j' is the index of
% data points, and there are J data points. 'k' is the index of point
% parameters, and there are K point parameters. 'l' is the index of
% distributed processes, and there are L distributed processes. 'ml' is the
% index for the m-th basis function of the l-th distribution. 'n' is the
% index of Monte-Carlo samples.

% Control the randomization. This helpful for debugging the code, but
% should be deleted in the implementation. 
rng(0)

% Clear the workspace, clear the command window, close all figures
clear;clc;close('all')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the data. The data should be a Jx3 matrix, where J is the number of
% data points. The data should be in Nyquist format, meaning that the 1st
% column is the vector of angular frequencies, the 2nd column is the vector
% of the real part of the impedance, and the 3rd column is the vector of
% the imaginary part of the impedance.
load('data_case_1.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Provide initial guesses. There are three initial guesses: point
% parameters, distributed parameters, and log-measurement variance. Point
% parameters are scalars. Distributed parameters have some mass and some
% characteristic timescale. The log-measurement variance is the natural
% log of the square of the relative error of the measurement. 

% The initial guess for the point parameters (betak) is a vector of size
% Kx1. The initial guess for the distributed parameters (Rtaul) is a matrix
% of size Lx2, where L is the number of processes. For example, for the
% Randles circuit, the number of processes is 2. The first column consists
% of the resistances, the second column consists of the characteristic
% timescales. For convenience, the definition of each initial guess is
% shown: 
Rinf=1;R1=1;tau1=1;
betak=Rinf;
Rtaul=[R1,Rinf];

% The measurement error is assumed to be of the form:
%    Z = Zhat + e*abs(Zhat)*(N(0,1)+1i*N(0,1)
% Here Z is the measured impedance and Zhat is the true underlying
% impedance. The initial guess for the log-measurement variance is:
%    mue ~ ln(e^2)
mue=-10.5;

% If any of the initial guess is not known, replace it with NaN. This is
% not recommended for very complicated models.

% Specify the nature of the distributions in the same order as the initial
% guesses and the input model. For EIS case 1, the underlying model is a
% series relaxation process, so there is only one characteristic timescale.
distType=cell(1);
distType{1}='series';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run invertEIS.m. The outputs are rather particular. 'modality' is a
% vector of size Lx1 whose l-th element refers to the number of basis
% functions used to estimate the l-th process (Ml). 'betak' is a matrix of
% size Kx3. The first column of betak is the lower bound estimate, the
% second column is the maximum likelihood estimate, the third column is the
% upper bound estimate. The bounds are 95% credible intervals. 'Rml' is a
% matrix of size (M1+M2+...)x3, where (M1+M2+...) is the total number of
% all basis functions. 'Rml' contains the masses of all the basis
% functions. The three columns are, as before, the lower bound, maximum
% likelihood, and the upper bound estimate respectively. 'muml' and 'wml'
% are identical to 'Rml', except they refer to means and log-variances of
% all the basis functions.
%
% The next set of outputs are 'betakn','Rmln','mumln', and 'wmln'. These
% are the Monte-Carlo samples leading to 'betak', 'Rml', 'muml', and 'wml',
% respectively. I suspect that someone may be interested in calculating
% derived quantities from the aforementionted quantities. The simplest way
% to do this is to resample from these Monte-Carlo samples. 
%
% The next output is 'wen'. This is the Monte-Carlo samples of the
% log-measurement variance.
%
% The next set of outputs are 'tl' and 'Fl'. Both are cells of size Lx1,
% describing the l-th distribution characteristic timescales. Each element
% of Fl consists of three columns, corresponding to the lower bound,
% maximum likelihood, and upper bound estimate respectively. 
[modality,betak,Rml,muml,wml,...
    betakn,Rmln,mumln,wmln,...
    wen,...
    tl,Fl]=invertEIS(@myFun,data,distType,betak,Rtaul,mue);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The DRT model maybe used in lieu of a user-defined function. This
% provides an elementary analysis suggesting the number, size, and
% characteristic timescales of distributed processes.
[modality,betak,Rml,muml,wml,...
    betakn,Rmln,mumln,wmln,...
    wen,...
    tl,Fl]=invertEIS(@DRT,data);

% We have provided four elementary models: @DRT, which returns the
% distribution of relaxation time, @transmissiveDDT, which returns the
% distribution of diffusion time for transmissive planar boundary
% condition, @infiniteDDT, which returns the distribution of diffusion time
% for a semi-infinite planar boundary condition, and @blockingDDT, which
% returns the distribution of diffusion time for blocking planar boundary
% condition.
% 
% Description of the DDT boundary conditions can be found in Song and
% Bazant (2018).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We compare the inversion result with the true underlying distribution.
% Extract the first distribution.
FlTemp=Fl{1};

% Plot the first distribution.
figure(1)
plot(t,F1,'k','LineWidth',1);hold('on')
plot(tl{1},FlTemp(2,:),'r','LineWidth',1)
plot(tl{1},FlTemp(1,:),'r-.','LineWidth',1)
plot(tl{1},FlTemp(3,:),'r-.','LineWidth',1)

% Label the first distribution.
xlabel('t')
ylabel('F_1(t)')
legend('True Distribution','Inversion Output')