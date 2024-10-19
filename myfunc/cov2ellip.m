function [ellipse,axesL,V] = cov2ellip(sig,k,c,resol)
% sig: sigma, cov matrix. in 1d, sigma is var, not standard diviation
% k: radius. in 1d, k*sqrt(sigma) is 99.7% of the data
% c: center of ellips
% resol: resolution, in 3d, theta resolution and phi resolution

d = size(c,1);

[V, E] = eig(sig); % rotation matrix and eigen value
if det(sig)<=1e-6
    V = abs(V); E = abs(E);
    % when det(sig) is very small, matlab will get img number even it is positive definite
    % So take abs value
end
axesL = k * sqrt(diag(E));

if ~isscalar(resol) && d==2
    resol = resol(1);
elseif isscalar(resol) && d==3
    resol = [resol,resol];
end

if d==2
    theta = linspace(0, 2*pi, resol);
    ellipse = V*(axesL.*[cos(theta); sin(theta)]) + c;
elseif d==3
    % create a sphere
    phi = linspace(-pi/2, pi/2, resol(1))';
    theta = linspace(0, 2*pi, resol(2)+1);
    x = cos(phi).*cos(theta);
    y = cos(phi).*sin(theta);
    z = sin(phi).*ones(1,resol(2)+1);

    % transform to ellips
    X = V * (axesL .* [x(:),y(:),z(:)]') + c;
    outsize = [resol(1),resol(2)+1];
    ellipse = NaN(outsize(1),outsize(2),3);
    for j = 1:3
        ellipse(:,:,j) = reshape(X(j,:),outsize);
    end
end

% sort axesL and V in order
[axesL,ind] = sort(axesL,'descend');
V = V(:,ind);

end