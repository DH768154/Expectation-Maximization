function dist = DistAB(A,B,varargin)
% dist = DistAB(A,B,steps,distfunc)
%
% Distance between 2 Column Vector Array with dimention d
% Designed for Large Matrix since my laptop don't have large memory.
% 
% If need L2 Distance, L2AB(A,B) function used 
% 
% -------------------------------------------------------------------
% Input:
%
% A = [a1 , a2 , ... , an]; % d*n matrix
% B = [b1 , b2 , ... , bm]; % d*m matrix
%
% steps = [stepA , stepB]
% step control the calculation matrix size, matrix size for each
% calculation will be: d*stepA*stepB.
%
% distfunc is self defined distant function. 
% Example, L2 norm:
% distfunc =@(a,b) sum((a - b).^2,1);
% Example, L1 norm:
% distfunc =@(a,b) sum(abs(a - b),1);
%
% -------------------------------------------------------------------
% Output:
%
% dist = [a1~b1 , a2~b1 , ...an~b1;
%         a1~b2 , a2~b2 , ...an~b2;
%         ........................;
%         a1~bm , a2~bm , ...an~bm]
%
% -------------------------------------------------------------------
% v1.0 | 10-05-2024 | Ding,Hao-sheng
% v1.1 | 10-06-2024 | Add self defined distance function. Create new L2AB
% function for L2 norm and speed up calculation.
% v1.2 | 10-09-2024 | L2AB allow using small size matrix
% v1.3 | 10-10-2024 | Merge L2AB into DistAB

%%
p = inputParser;
addParameter(p,'DistFunc',[]);
addParameter(p,'StepSize',inf);
addParameter(p,'DispIter',false);
parse(p,varargin{:});
distfunc = p.Results.DistFunc;
stepsize = p.Results.StepSize;
DispIter = p.Results.DispIter;

%% if use L2 Norm, apply L2AB Function

if isempty(distfunc)
    dist = L2AB(A,B,'StepSize',stepsize,'DispIter',DispIter);
    return
end

%%
[dim,nA] = size(A);
nB = size(B,2);

%% If not L2 Norm, and steps is inf, use 3D matrix Calculate Dirctly
if all(isinf(stepsize))
    dist = distfunc(A,reshape(B,dim,1,nB));
    dist = reshape(dist,[nA,nB])';
    return
end

%% One of the Step is non inf and dist Function is defined

stepsize = stepsize.*[1,1];
if isinf(stepsize(1))
    niterA = 1;
    stepsize(1) = nA;
else
    niterA = ceil(nA/stepsize(1));
end

if isinf(stepsize(2))
    niterB = 1;
    stepsize(2) = nB;
else
    niterB = ceil(nB/stepsize(2));
end

B = reshape(B,dim,1,nB);
dist = NaN(1,nA,nB);
for j = 1:niterB
    cindB = ((j-1)*stepsize(2)+1):min([(j*stepsize(2)),nB]);
    Bc = B(:,1,cindB);
    for i=1:niterA
        cindA = ((i-1)*stepsize(1)+1):min([(i*stepsize(1)),nA]);
        dist(1,cindA,cindB) = distfunc(A(:,cindA),Bc);
        if DispIter
            fprintf('Loop (%3.0f , %3.0f) in (%3.0f , %3.0f)\n',i,j,niterA,niterB);
        end
    end
end

dist = reshape(dist,[nA,nB])';
end