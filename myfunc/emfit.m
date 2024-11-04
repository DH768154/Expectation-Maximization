function gm = emfit(O,m0,s0,p_k0,varargin)
% d dimention, n_set cluster, n_data data points
% m0: initial mean, size = [d,n_set]
% s0: initial sigma, size = [d,d,n_set]
% p_k0: initial mixing coefficients, size = [1,n_set]
% 
% Optional Input:
% tol: stop criteria, average(4) change of avg log likelihood
% dispinfo: show figure and print
% alldata: all data for each iteration
%
% You Can Use ChatGPT or Goole Translate to Understand the Chinese Commend.
% 
% 后续代码注释以红蓝两个符合高斯分布的事件为例，比如红蓝两把枪射击100次，但这个代码不仅限于分2类。
% EM算法旨识别出这两组高斯事件：红枪和蓝枪分别打出的中心位置和协方差（协方差决定数据分布椭圆形状）。
%
% O是这100次射击在一面墙上留下的坐标。
%
% p_k代表这100次里用红枪和蓝枪射击的概率，如红色打了10次，蓝色打90次，那p_k就是0.1和0.9。
%
% 红枪和蓝枪打靶都符合高斯分布，中间数量多周围数量少。因此，如果知道这两组高斯分布的参数（中心位置
% 和协方差），给定墙上的一个坐标，假定只有一把枪射击（只有一个高斯分布），可以计算出这个坐标的概率。
% 即如果靠近这把枪高斯分布的中心，概率很高，远离中心，概率很小。
% 分别计算数据中每个点位在红蓝两种情况下的概率，再用红和蓝的p_k分别给这两组概率加权，就是p_k_x。
% 
% v0.1 | 10-15-2024 | DH768154
% v0.2 | 10-15-2024 | output all iteration results
% v1.0 | 10-16-2024 | add stop criteria
% v1.1 | 10-17-2024 | fix problem if one point is far away from all center, a/0 = nan
% v1.2 | 10-17-2024 | add regularize term when cov matrix is close singular
% v1.3 | 10-18-2024 | if each element in cov is close to zero, delete this gauss
% v1.4 | 10-19-2024 | add another check, if many points are at same location and far away 
% from other data, do not delete this gauss
% v1.5 | 10-24-2024 | change stop criteria to change of avg log likelihood,
% plot , add save all iter data option
%%
p = inputParser;
addParameter(p,'maxiter',300)
addParameter(p,'tol',1e-5)
addParameter(p,'dispinfo',true);
addParameter(p,'alldata',false);
parse(p,varargin{:});
maxiter = p.Results.maxiter;
tol = p.Results.tol;
dispinfo = p.Results.dispinfo;
alldata = p.Results.alldata;

%% Size

[d,n_data] = size(O);
n_set = size(m0,2);

%% Initial

p_k0 = p_k0/sum(p_k0);

% resize to 3d, vectorize calculation
p_k = reshape(p_k0,1,1,[]);
m = reshape(m0,d,1,[]);
s = s0;

count = maxiter;

%% record all data

if alldata
    m_all = NaN(d,maxiter+1,n_set);
    s_all = NaN(d,d,n_set,maxiter+1);
    pk_all = NaN(1,maxiter+1,n_set);

    m_all(:,1,:) = m;
    s_all(:,:,:,1) = s;
    pk_all(:,1,:) = p_k;
end

log_likelihood = NaN(1,maxiter+1);

%%

if dispinfo
    f = figure;
    pp = plot(nan,'.-','LineWidth',1);
    grid on; xlim([1,inf])
    title('avg log likelihood')
    set(f,'Units','normalized','Position',[0.2,0.2,0.6,0.6])
end

%%

ind_keep = 1:n_set;
for i = 1:maxiter

    %% E-step: Compute responsibilities (posterior probabilities)

    % p(k|x): 给一个数，计算这个数分别在红蓝两种情况下发生的概率
    % p(k|x) size: [1,n_data,n_set]
    p_k_x = p_k.* ndgauss(O,m,s);

    % 比如观测到的值是10，分别计算红色或者蓝色得到10的概率。
    % 因为这两个概率之和是1，所以要除以sum(p_k_x,3)。
    p_x = sum(p_k_x,3);
    likelihood = mean(log(p_x)); % avg log likelihood
    p_k_x = p_k_x./p_x; 
    

    % 但是由于这个点太远了，红色和蓝色的高斯分布曲线算出来的结果都很接近于0，计算机认为就是0。
    % 除以0会变成nan，因此重新调整概率，认为红蓝概率相等，0.5，0.5
    ind = p_x==0;    
    p_k_x(:,ind,:) = 1/n_set;

    %% M-step: Update parameters

    % 计算红和蓝发生的概率
    Nk = sum(p_k_x,2); 
    p_k = Nk / n_data; % mixing coefficients   

    % 临时变量，使得p_k_x里红色的（或蓝色）概率加起来是1
    p_k_x = p_k_x ./ Nk; % temp var, not p_k_x
    p_k_x(:,:,Nk==0) = 1/n_data; % avoid one center is too far away
    
    % Update center, weighted avg, the temp var in pre step is weight  
    m = sum(O.*(p_k_x),2);

    % update cov matrix
    % 假设每一组（红或蓝）去中心化的数据为Xc，这一组对应的p_k_x为p_ki_x
    % 新的协方差为：(p_ki_x .* Xc) * Xc'
    % 数据集中每一个数据与自己的协方差，以p_ki_x加权以后全部加起来
    % 如Xc = [xc1,xc2...,xcn]
    % 新的协方差为: p_ki_x(1)*xc1*xc1' + p_ki_x(2)*xc2*xc2' +...+ p_ki_x(n)*xcn*xcn'
    s = pagemtimes(p_k_x .* (O-m),pagetranspose(O-m));


    % 如果协方差矩阵所有元素接近于0，代表这个高斯分布椭圆半径几乎为0，数据在这个高斯分布中概率极小
    % 可能是因为所有点都离这个高斯分布中心很远，但也有可能很多点有同一个坐标。
    % 前者可以直接把这个高斯分布删除，后者不行。可以用红蓝概率p_k检测排除后者。
    ind_r1 = all(abs(s)<=1e-6,[1,2]); % check cov matrix almost 0
    ind_r2 = p_k<1/n_set/100; % check not too many points at the same location
    ind_r = all([ind_r1,ind_r2]);
    if any(ind_r)
        ind_keep = ind_keep(~ind_r);
        n_set = length(ind_keep);
        s = s(:,:,~ind_r);
        m = m(:,:,~ind_r);
        p_k = p_k(:,:,~ind_r);
    end

    % Regularize Sigma if det(Sigma) is too small
    % to ensure positive definiteness
    ind_det0 = arrayfun(@(j) det(s(:,:,j)),1:n_set) <= 1e-6;
    s(:,:,ind_det0) = s(:,:,ind_det0) + eye(d)*(1e-6);

    %% record all data

    if alldata
        m_all(:,i+1,ind_keep) = m;
        s_all(:,:,ind_keep,i+1) = s;
        pk_all(:,i+1,ind_keep) = p_k;
    end
    log_likelihood(i) = likelihood;

    %% if log-likelihood not change, break loop
    % log-likelihood不再变化，跳出循环

    if i>=5
        d_likelihood = abs(mean(diff(log_likelihood(i-4:i))));
        %d_likelihood = likelihood_pre-likelihood;
        c0 = abs(d_likelihood)<tol;
    else
        d_likelihood = NaN;
        c0 = false;
    end
    
    if (mod(i,10)==0 || i==maxiter || c0) && dispinfo
        fprintf('iter: %3.0f ~%3.0f | change = %11.3e\n',max(i-4,1),i,d_likelihood)

    end
    if dispinfo
        pp.YData = log_likelihood(1:i);
        pp.XData = 1:i;
        drawnow; 
    end

    if c0
        count = i;
        break
    end
end

%% squeeze final output

m = squeeze(m);
p_k = squeeze(p_k);

%% record all parameter for plotting
if alldata
    m_all = m_all(:,1:count+1,:);
    s_all = s_all(:,:,:,1:count+1);
    pk_all = pk_all(:,1:count+1,:);
end
log_likelihood = log_likelihood(1:count+1);

%%
gm.m = m;
gm.s = s;
gm.p_k = p_k;
gm.log_likelihood = log_likelihood;
if alldata
    gm.m_all = m_all;
    gm.s_all = s_all;
    gm.pk_all = pk_all;
end
end