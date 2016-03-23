function [gmm,resp,obj,iter_time] = soft_kmeans(X,K)
%% soft k-means
rng('default');
options.display_plots = 0;

%% generate data
if (nargin < 1)
dataset = 'iris'; %'faithful', 'mnist'

if (strcmp(dataset,'faithful'))
load faithful
X = faithful;

%standardize data
Xmean = mean(X,1);
Xsigma = std(X,0,1);

[n,d] = size(X);
X = X - repmat(Xmean,n,1);
X = X./repmat(Xsigma,n,1);

if (options.display_plots)
figure;
scatter(X(:,1),X(:,2));
end

elseif strcmp(dataset,'iris')

load fisherIrisData.mat
X = meas;
trueK = length(unique(species));

[n,d] = size(X);

%standardize data
Xmean = mean(X,1);
Xsigma = std(X,0,1);

X = X - repmat(Xmean,n,1);
X = X./repmat(Xsigma,n,1);

%compute pca
[x_coeff, x_score] = pca(X);
X_pca = x_score(:,1:2);
if(options.display_plots)
figure;
scatter(X_pca(:,1),X_pca(:,2));
xlabel('PC 1'); ylabel('PC 2'); title('IRIS PCA Plot');
end
     
elseif (strcmp(dataset,'mnist'))

class = [2 3]; %binary classification
%[Xtrain, ytrain, Xtest, ytest] = mnistLoad(class);
[X, ~, ~, ~] = mnistLoad(class);

[n,d] = size(X);

%compute pca
[x_coeff, x_score] = pca(X);
X_pca = x_score(:,1:2);

if(options.display_plots)
figure;
scatter(X_pca(:,1),X_pca(:,2));
xlabel('PC 1'); ylabel('PC 2'); title('MNIST PCA Plot');

figure;
idx = randperm(n); idx = idx(1:64);
display_mnist(X(idx,:));
end

end
end

%% kmeans parameters
%K = 2;
[n,d] = size(X);

gmm.K = K;
gmm.d = d;
gmm.z = mod(randperm(n),K)+1;
gmm.mu = randn(K,d);
gmm.sigma = sparse(2^2*eye(d));
gmm.pik = 1/K*ones(K,1);
gmm.nk = zeros(K,1);
gmm.beta = 1; %used in exp(-beta * dist(xi,xj))


%% k++ init

gmm.mu = kpp_init(X,gmm.K);

%% soft k-means

max_iter = 10;
dist = zeros(n,gmm.K);
resp = zeros(n,gmm.K);
obj = zeros(max_iter,1);
iter_time = zeros(max_iter,1);

obj_tol = 5;
Np = [1 2 3 max_iter]; cnt = 1;

for iter = 1:max_iter
    tic;
    
    %assignment step
    for kk = 1:gmm.K
        dtemp = X - repmat(gmm.mu(kk,:),n,1);        
        dist(:,kk) = sum(dtemp.*dtemp,2); 
    end
    [resp] = soft_max(dist,gmm.beta);

    %update labels
    [~,gmm.z] = min(dist,[],2);
    
    %update step
    for kk = 1:gmm.K
        gmm.mu(kk,:) = sum(repmat(resp(:,kk),1,size(X,2)).*X,1)./repmat(sum(resp(:,kk),1),1,size(X,2)); 
        gmm.nk(kk) = sum(gmm.z == kk);
    end        
    
    %compute objective
    for kk = 1:gmm.K
       obj(iter) = obj(iter) + sum(dist(gmm.z == kk,kk));
    end

    %check convergence
    if (iter > 1 && abs(obj(iter) - obj(iter-1)) < obj_tol)
        break;
    end
    
    %generate plots
    if (iter == Np(cnt) && options.display_plots)
        cnt = cnt+1;
        figure;
        for kk = 1:gmm.K
            scatter(X_pca(gmm.z==kk,1),X_pca(gmm.z==kk,2)); hold on;
        end
        xlabel('PC 1'); ylabel('PC 2'); title(['iter: ',num2str(iter)]);
    end
    iter_time(iter) = toc;
end

%% generate plots

if (options.display_plots)

figure;
for kk = 1:gmm.K
    scatter(X_pca(gmm.z==kk,1),X_pca(gmm.z==kk,2)); hold on;
end
xlabel('PC 1'); ylabel('PC 2');

if (strcmp(dataset,'mnist'))
for kk = 1:gmm.K
    idx = find(gmm.z == kk);
    idx_temp = randperm(length(idx)); idx = idx_temp(1:64);
    figure;
    display_mnist(X(idx,:));
end
end

figure;
plot(obj,'-b','linewidth',1.0); xlabel('iterations'); ylabel('sum of l2 squared distances');
title('k-means objective');

end


end

function [mu] = kpp_init(X,k)
%k++ init

[n,d] = size(X);
mu = zeros(k,d);
dist = inf(n,1);

mu(1,:) = X(ceil(rand*n),:);
for i = 2:k
    D = bsxfun(@minus,X,mu(i-1,:));
    dist = min(dist,dot(D,D,2));    
    idx = find(rand < cumsum(dist/sum(dist)),1);
    mu(i,:) = X(idx,:);
end

end

function [r] = soft_max(dist,beta)

[n,k] = size(dist);
r = zeros(n,k);

for kk = 1:k
   r(:,kk) = exp(-beta*dist(:,kk));
end
%normalize
r = r./repmat(sum(r,2),1,k);

end

%% notes
