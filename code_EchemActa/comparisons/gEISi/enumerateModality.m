function modality=enumerateModality(optModality)

% MODALITY = ENUMERATEMODALITY(OPTMODALITY) creates a list of optimization
% tasks given the current best modality. The optimization tasks are
% generated heuristically. For example, given that the current optimal
% modality is [2;1], we should move on to investigate [3;1] and [2;2].
% 
% input:
% (1) optModality: size: Lx1
%                  class: double
%     definition: L is the number of processes
%     description: vector of the current best number of basis functions
%                  needed to estimate the l-th distribution
% 
% output:
% (1) modality: size: LxL
%               class: double
%     description: a list of optimization tasks; the l-th column
%                  corresponds to the l-th task
% 
% Author: Surya Effendy
% Date: 08/03/2019

% Declare globals
global L

% Case #1: current optimal modality is empty i.e. initialization
if isempty(optModality)
    modality=ones(L,1);
else
    % Case #2: given current optimal modality, we step up the complexity of
    % each distribution by 1
    modality=bsxfun(@plus,optModality,eye(L));
end

end