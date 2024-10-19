function Y = ndgauss(x,m,s)
[d,~,k] = size(m);
k2 = size(k,3);
if k2~=k2
    error('dimention not agree')
end
Y = NaN(1,size(x,2),k);
for i = 1:k
    if det(s(:,:,i))<1e-6
        s(:,:,i) = s(:,:,i)+eye(d)*(1e-6);
    end
    scale = 1 / sqrt((2*pi)^d * det(s(:,:,i)));
    Y(:,:,i) = scale * exp(-sum(s(:,:,i)\(x-m(:,:,i)) .* (x-m(:,:,i)), 1)/2);
end


end