function tl=generateMesh(muml,wml)

% TL = GENERATEMESH(MUML,WML) generates dimensionless mesh node
% quadratically about the mean of each distribution. This heuristic assumes
% that kernel (K) is bounded somehere inside the inverse square root of the
% machine precision i.e. if my model takes the form:
%     Z=Kf
% Then each element of K must be within +67108864 and -67108864. The idea
% behind this heuristic is that for such kernels, sufficiently small values
% of f can be neglected. Let
%     z=(lml-muml)/exp(wml/2)
% Here muml is the mean of the ml-th distribution, and wml is the
% corresponding log-variance of the distribution. The standard normal
% distribution falls below the machine precision when
%     eps<1/sqrt(2*pi)*exp(-z^2)
% It follows that
%     z<sqrt(-ln(sqrt(2*pi)*eps) & z>-sqrt(-ln(sqrt(2*pi)*eps)
% 
% input:
% (1) muml: size: Mlx1
%           class: double
%     definition: Ml is the number of basis functions in the l-th
%                 distribution
%     description: vector of basis function means
% (2) wml: size: Mlx1
%          class: double
%     description: vector of basis function log-variances
% 
% output:
% (1) tl: size: (49*Ml)x1
%         class: double
%     description: vector of mesh points for the l-th process
% 
% Author: Surya Effendy
% Date: 03/08/2019

% Let hv be the nondimensional mesh points
hv=[-5.92661073936128,-5.44301576583701,-4.97999930182441,...
    -4.53756134732348,-4.11570190233423,-3.71442096685664,...
    -3.33371854089072,-2.97359462443648,-2.63404921749390,...
    -2.31508232006300,-2.01669393214377,-1.73888405373621,...
    -1.48165268484032,-1.24499982545610,-1.02892547558356,...
    -0.833429635222680,-0.658512304373476,-0.504173483035943,...
    -0.370413171210080,-0.257231368895889,-0.164628076093369,...
    -0.0926032928025200,-0.0411570190233422,-0.0102892547558356,...
    0,...
    0.0102892547558356,0.0411570190233422,0.0926032928025200,...
    0.164628076093369,0.257231368895889,0.370413171210080,...
    0.504173483035943,0.658512304373476,0.833429635222680,...
    1.02892547558356,1.24499982545610,1.48165268484032,...
    1.73888405373621,2.01669393214377,2.31508232006300,...
    2.63404921749390,2.97359462443648,3.33371854089072,...
    3.71442096685664,4.11570190233423,4.53756134732348,...
    4.97999930182441,5.44301576583701,5.92661073936128];

% Calculate dimensional mesh points for each ml
lml=bsxfun(@plus,bsxfun(@times,hv,exp(wml/2)),muml);
tl=sort(lml(:),'Ascend')';

end