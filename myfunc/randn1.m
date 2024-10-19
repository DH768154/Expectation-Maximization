function X = randn1(s,n,c)
% s: cov matrix
% n: n points
% c: center
X = chol(s,'lower') * randn(size(s,1),n) + c;
end