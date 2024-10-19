function C = getSeeds(X,k,useMax,varargin)

if isempty(varargin{1})
    distfunc = @(a,b) sum((a-b).^2,1);
else
    distfunc = varargin{1};
end

[dim,n] = size(X);
C = NaN(dim,k);

%% Select 1st Seed
ind = randi(n);
C(:,1) = X(:,ind);
D = distfunc(C(:,1),X); % Distance

for i = 2:k

    if useMax % Use Extreme Value
        [~,ind] = max(D);
    else
        % Select next seed based on probability
        P = D/sum(D); % probability
        cumP = cumsum(P); % PDF
        ind = find(cumP>=rand(1),1);
    end
    
    C(:,i) = X(:,ind);

    % Update min distance to seeds
    D = min(D,distfunc(C(:,i),X));
end
end