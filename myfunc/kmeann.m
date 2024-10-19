function [idx,C,S,iters] = kmeann(X,k,maxiter,ntrial,varargin)
p = inputParser;
addParameter(p,'DistFunc',[]);
addParameter(p,'Seed','++');
addParameter(p,'DispIter',false);
parse(p,varargin{:});
distfunc = p.Results.DistFunc;
seed = p.Results.Seed;
DispIter = p.Results.DispIter;

%%
S = inf;
for i = 1:ntrial

    if strcmpi(seed,'rand')
        C_int = X(:,randperm(size(X,2), k));
    elseif strcmpi(seed,'maxdist')
        C_int = getSeeds(X,k,true,distfunc);
    else
        C_int = getSeeds(X,k,false,distfunc);
    end

    [idxc,Cc,Sc,count] = k_mean(X,k,C_int,maxiter,distfunc);

    sc_all = sum(Sc);
    if sc_all < S
        S = sum(Sc);
        C = Cc;
        idx = idxc;
        iters = count;
    end
    if DispIter
        fprintf('trial %2.0f: std = %4.4f | iter = %3.0f\n',i,sc_all,count)
    end
end
end

%%
function [idx,C,s,count] = k_mean(X,k,C_int,maxiter,distfunc)


idx_p = zeros(1,size(X,2));
count = maxiter;

for i = 1:maxiter
    if i==1
        C = C_int;
    end

    dist = DistAB(X,C,'DistFunc',distfunc);
    [~,idx] = min(dist,[],1);

    for j = 1:k
        C(:,j) = mean(X(:,idx==j),2);
    end

    if idx == idx_p
        count = i;
        break
    end
    idx_p = idx;

end

s = NaN(1,k);
for i = 1:k
    s(i) = std(dist(i,idx==i));
end
end