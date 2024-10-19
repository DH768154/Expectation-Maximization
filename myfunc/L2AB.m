function dist = L2AB(A,B,varargin)
% L2 norm^2 between 2 Column Vector Array with dimention d
% 
% Input:
%
% A = [a1 , a2 , ... , an]; % d*n matrix
% B = [b1 , b2 , ... , bm]; % d*m matrix
%
% Output:
%
% dist = [a1~b1 , a2~b1 , ...an~b1;
%         a1~b2 , a2~b2 , ...an~b2;
%         ........................;
%         a1~bm , a2~bm , ...an~bm]
%
% 10-10-2024 | Ding,Hao-sheng
p = inputParser;
addParameter(p,'StepSize',inf);
addParameter(p,'DispIter',false);
parse(p,varargin{:});
stepsize = p.Results.StepSize;
DispIter = p.Results.DispIter;

if isinf(stepsize)
    dist = sum(A.^2) + sum(B.^2).' -2*B.' * A;
else
    nB = size(B,2);
    nA = size(A,2);
    nstep = ceil(nB/stepsize);
    dist = NaN(nB,nA);
    for i = 1:nstep
        ind = ((i-1)*stepsize+1) : min([nB,i*stepsize]);
        Bp = B(:,ind);
        dist(ind,:) = sum(A.^2) + sum(Bp.^2).' -2*Bp.' * A;
        if DispIter
            fprintf('Loop %3.0f / %3.0f\n',i,nstep);
        end
    end
end



end