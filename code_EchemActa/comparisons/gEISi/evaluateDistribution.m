function [tl,Fl]=evaluateDistribution(Rml,muml,wml,distType,varargin)

% [TL,FL] = EVALUATEDISTRIBUTION(RML,MUML,WML,DISTTYPE,TL) evaluates the
% l-th distribution at pre-specified mesh points. If tl is provided as
% input, then Fl will be evaluated at the specified values of tl.
% Otherwise, tl will be generated quadratically about the means of the
% basis functions.
% 
% input:
% (1) Rml: size: Mlx1
%          class: double
%     definition: Ml is the number of basis functions needed to approximate
%                 the l-th distribution
%     description: vector of basis function masses
% (2) muml: size: Mlx1
%           class: double
%     description: vector of basis function means
% (3) wml: size: Mlx1
%          class: double
%     description: vector of basis function log-variances
% (4) distType: size: 1x1
%               class: char
%     description: describes the distributed nature of the process; can be
%                  either 'series' or 'parallel'
% (5) tl: size: Ox1
%         class: double
%     definition: O is the number of mesh points
%     description: vector of mesh points for the distribution; keep empty
%                  to auto-generate the mesh points
% 
% output:
% (1) tl: size: Ox1
%         class: double
%     description: vector of mesh points under consideration
% (2) Fl: size: Ox1
%         class: double
%     description: vector of l-th distribution evaluated at tl
% 
% Author: Surya Effendy
% Date: 03/12/2019

% Evaluate t, if necessary.
if size(varargin,1)==1
    tl=varargin{1};
else
    tl=generateMesh(muml,wml);
end

% Evaluate the distributions.
if strcmp(distType,'series')
    Fl=bsxfun(@times,Rml/sqrt(2*pi)./exp(wml/2),...
        exp(-bsxfun(@rdivide,bsxfun(@minus,tl,muml).^2,2*exp(wml))));
    Fl=sum(Fl,1);
elseif strcmp(distType,'parallel')
    Fl=bsxfun(@times,1./Rml/sqrt(2*pi)./exp(wml/2),...
        exp(-bsxfun(@rdivide,bsxfun(@minus,tl,muml).^2,2*exp(wml))));
    Fl=sum(Fl,1);
end

end