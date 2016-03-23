function [gmm,obj,resp,iter_time] = gmm_em(X,K)
%% Gaussian Mixture Model using EM

rng('default')
options.display_plots = 0;

%% generate data
if (nargin < 1)

n=1e4;
d=2;
K=4;

[X,mu0,V0] = gen_data(n,d,K);

if (options.display_plots)
figure;
scatter(X(:,1),X(:,2)); hold on; grid on;
for k=1:K, plot2dgauss(mu0(:,k), V0(:,:,k)); hold on; end
title('ground truth');
end

end

%% gmm parameters

[n,d] = size(X);

gmm.K = K;
gmm.mu = rand(d,K);
gmm.sigma = repmat(eye(d,d),1,1,K);
gmm.pik = ones(K,1)/K;

%% k-means init

[gmm_kmeans] = kmeans(X,K);
gmm.mu = gmm_kmeans.mu';

%% EM algorithm

max_iter = 1e1;
tol = 1e-5;

obj  = zeros(max_iter,1);
iter_time = zeros(max_iter,1);
Np = [1 5 8 max_iter]; cnt = 1;

for iter = 1:max_iter
    tic;
    fprintf('EM iter: %d\n', iter);
    
    %Estep
    [resp, llh] = estep(X, gmm);

    %Mstep
    [gmm] = mstep(X,resp);

    %check convergence
    obj(iter) = llh;  
    if (iter > 1 && obj(iter)-obj(iter-1) < tol*abs(obj(iter)))
        break;
    end

    %generate plots
    if (options.display_plots && Np(cnt) == iter)
        cnt = cnt+1;
        figure;
        scatter(X(:,1),X(:,2)); hold on; grid on;
        for k=1:K, plot2dgauss(gmm.mu(:,k), gmm.sigma(:,:,k)); hold on; end
    end
    iter_time(iter) = toc;
end

%% generate plots

figure;
plot(obj); xlabel('iter'); ylabel('log-likelihood');
title('EM-GMM objective');

figure;
plot(iter_time); xlabel('iter'); ylabel('time, sec');
title('EM-GMM timing');


end

function [resp, obj] = estep(X, gmm)

K = gmm.K;
mu = gmm.mu;
sigma = gmm.sigma;
pik = gmm.pik;

[n,d] = size(X);
log_r = zeros(n,K);
for kk = 1:K 
    log_r(:,kk) = log_gauss_pdf(X, mu(:,kk), sigma(:,:,kk));  
end
log_r = bsxfun(@plus, log_r, log(pik)');
L = logsumexp(log_r,2);
obj = sum(L)/n; %log-likelihood
log_r = bsxfun(@minus, log_r, L); %normalize
resp = exp(log_r);

end

function [gmm] = mstep(X,resp)

d = size(X,2);
[n,k] = size(resp);

mu = zeros(d,k);
sigma = zeros(d,d,k);

nk = sum(resp,1);
pik = nk/n;
sqrt_resp = sqrt(resp);
for kk = 1:k
   rx = bsxfun(@times, resp(:,kk), X);
   mu(:,kk) = bsxfun(@times, sum(rx,1), 1./nk(kk));    
   
   Xm = bsxfun(@minus,X,mu(:,kk)');
   Xm = bsxfun(@times,Xm,sqrt_resp(:,kk));
   sigma(:,:,kk) = Xm'*Xm./nk(kk) + eye(d)*1e-6;
end

gmm.K = k;
gmm.pik = pik(:);
gmm.mu = mu;
gmm.sigma = sigma;

end

function logp = log_gauss_pdf(X, mu, Sigma)

d = size(Sigma, 2);
X  = reshape(X, [], d); 
X = bsxfun(@minus, X, mu(:)');
logp = -0.5*sum((X/(Sigma)).*X, 2); 
logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma);
logp = logp - logZ;
%p = exp(logp);        

end

function y = logdet(A)

try
    U = chol(A);
    y = 2*sum(log(diag(U)));
catch
    y = 0;
    warning('logdet:posdef', 'Matrix is not positive definite');
end

end

function s = logsumexp(a, dim)

if nargin < 2
  dim = 1;
end

% subtract the largest in each column
[y, i] = max(a,[],dim);
dims = ones(1,ndims(a));
dims(dim) = size(a,dim);
a = a - repmat(y, dims);
s = y + log(sum(exp(a),dim));
i = find(~isfinite(y));
if ~isempty(i)
  s(i) = y(i);
end

end


%% notes
