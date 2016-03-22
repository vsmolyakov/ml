function [] = mcmc_mh_lr()
%% Metropolis-Hastings for Bayesian Logistic Regression
% p(w) = N(w|m0, S0) prior
% log p(w|X,y) ~ log p(X|w,y) + log p(w) = -1/2(w-m0)'S0^-1(w-m0)
% + sum_{i=1}^{n} [y_i log mu_i + (1-y_i) log(1-mu_i)] + const (posterior)
% where mu_i = sigm(w'x_i)

options.display_plots = 0;

%% dataset

dataset = 'synthetic'; %'a9a';

if (strcmp(dataset,'synthetic'))
% synthetic data
n=1e3;

mu1 = [1,1]; mu2 = [-1,-1];
pik = [0.4, 0.6];

X = zeros(n,2); y = zeros(n,1);
for i=1:n
    u = rand; 
    idx = find(u<cumsum(pik),1,'first');
    if (idx == 1)
        X(i,:) = randn(1,2) + mu1;
        y(i) = 1;
    else
        X(i,:) = randn(1,2) + mu2;
        y(i) = -1;
    end         
end

elseif (strcmp(dataset,'a9a'))
% a9a dataset
%https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

DATA_PATH = './';
file_name_train = [DATA_PATH, 'a9a'];
%file_name_test = [DATA_PATH, 'a9a.t'];

[y, X] = libsvmread(file_name_train); 
%[y_train, X_train] = libsvmread(file_name_train);
%[y_test, X_test] = libsvmread(file_name_test); 

end

%add bias
%X = [ones(size(X,1),1) X];

n = size(X,1);
num_train = floor(0.8*n);
X_train = X(1:num_train,:); X_test = X(num_train+1:end,:);
y_train = y(1:num_train,:); y_test = y(num_train+1:end,:);

%% MAP estimate

d = size(X,2);
w0 = zeros(d,1);
lambda = 1e-4; %regularizer
w_map = irls_lr(X_train,y_train,lambda,w0);

mu = sigmoid(X_train*w_map);
Sk = diag(mu.*(1-mu));

%% distribution parameters

%prior
sigma0 = 4;
m0 = zeros(d,1); S0 = sigma0*eye(d);  %prior parameters
w0 = S0*randn(d,1) + m0; %sample w0 ~ N(m0,S0)

%proposal (init at w_map)
prop_params.type = 'gauss';
prop_params.mu = w_map;   %N(mu,sigma)
prop_params.sigma = (2.38^2/d)*inv(X_train'*Sk*X_train + inv(S0));

%target (posterior)
target_params.X = X_train;
target_params.y = y_train;
target_params.mu0 = m0;
target_params.Sigma0 = S0;
target_params.lambda = lambda;

%visualize target and proposal distributions
%{
[w1,w2] = meshgrid(1:0.1:4, 1:0.1:4); 
[nw,nw]=size(w1);
X=[reshape(w1,nw*nw,1), reshape(w2,nw*nw,1)];
p=zeros(size(X,1),1);
for i=size(X,1)
    p(i) = target(X(i,:),target_params);
end
q = proposal(X, prop_params);
figure; mesh(w1,w2,max(reshape(p,[nw,nw]),mean(p))); hold on;
mesh(w1,w2,reshape(q,[nw,nw]));
xlabel('x');ylabel('y');zlabel('p(x,y)')
figure; [cc,h]=contour(w1,w2,reshape(p,[nw,nw]),30); hold on;
[cc,h]=contour(w1,w2,reshape(q,[nw,nw]),30);
%}

%% MH algorithm

n = 5e3; %number of samples
num_chains = 3;
n_accept = zeros(num_chains,1);
s = zeros(n,d,num_chains);
burnin = floor(0.1*n);
alpha = zeros(n,1);

tic;
fprintf('starting MH sampling...\n');
for chain = 1:num_chains

rng(chain);
fprintf('chain: %d\n', chain);
s(1,:,chain) = sample_proposal(prop_params.mu,prop_params); %init

for i = 1:n
  x_curr = s(i,:,chain);
  x_new = sample_proposal(x_curr,prop_params);
  
  alpha(i) = proposal(x_new, prop_params) / proposal(x_curr, prop_params); % q(x|x')/q(x'|x)
  alpha(i) = alpha(i) * (target(x_new,target_params)/target(x_curr,target_params)); %p(x')/p(x)
  
  r = min(1,alpha(i));
  u = rand;
  if (u < r)
      s(i+1,:,chain) = x_new;  %accept
      n_accept(chain) = n_accept(chain) + 1;
  else
      s(i+1,:,chain) = x_curr; %reject
  end
    
end
end
time = toc;
fprintf('elapsed time: %.4f sec\n', time);
fprintf('avg acceptance ratio: %.4f\n', mean(n_accept)/n);

%% convergence diagnostic

if(options.display_plots)
figure;
plot(s(:,:,1)); hold on;
plot(s(:,:,2)); hold on;
plot(s(:,:,3)); hold on;
title('trace plot');
xlabel('number of samples'); ylabel('samples');
end

[R,neff,Vh,W,B,tau,thin] = psrf(s(:,:,1),s(:,:,2),s(:,:,3));

%thin and merge samples after burnin
samples = [s(burnin+1:end,:,1);s(burnin+1:end,:,2);s(burnin+1:end,:,3)];
%samples = [s(burnin+1:thin:end,:,1);s(burnin+1:thin:end,:,2);s(burnin+1:thin:end,:,3)];

%SAVE_PATH = './';
%save([SAVE_PATH,'mcmc_mh_lr.mat']);

%% generate plots

w_post_mean = mean(samples,1)'
y_pred = 2*(sigmoid(X_test*w_post_mean) >= 0.5)-1;
y_err = length(find(y_pred-y_test))/length(y_test);
fprintf('classification error: %.4f\n', y_err);

%compute pca
[w_coeff, w_score] = pca(samples);
w_pca = w_score(:,1:2);
figure; scatter(w_pca(:,1), w_pca(:,2));
xlabel('PC 1'); ylabel('PC 2'); title('MCMC-MH Posterior PCA Plot');

if(options.display_plots)
figure;
scatter(samples(:,1),samples(:,2));
xlabel('w1'); ylabel('w2'); legend('p(w|x,y)'); title('MCMC-MH posterior');

figure;
scatter(X(find(y==1),1),X(find(y==1),2), 'ro'); hold on;
scatter(X(find(y==-1),1),X(find(y==-1),2), 'bo'); hold on;
x1 = linspace(min(X(:,1))-1,max(X(:,1))+1,10);
plot(x1, -(w_post_mean(1)/w_post_mean(2))*x1, '-r');
xlabel('x1'); ylabel('x2'); title('baesian logistic regression');
end

end


function p = target(w,target_params)
% log P(theta|x) = N(w0,S0) prod_i mu_i^y_i x (1-mu_i)^(1-y_i)

X = target_params.X;
y = target_params.y;
m0 = target_params.mu0;
S0 = target_params.Sigma0;

d = size(X,2);
w = reshape(w,d,[]);

y01 = (y+1)/2;

mu = sigmoid(X*w);
mu = max(mu,eps);    %bound away from 0
mu = min(mu,1-eps);  %bound away from 1

logp =  sum(y01.*log(mu)+(1-y01).*log(1-mu)) + log(gauss_pdf(w,m0,S0)); 
p = exp(logp);

end

function p = proposal(X, prop_params)
% Multivariate Gaussian distribution, pdf

type = prop_params.type;

if (strcmp(type,'gauss'))
    mu = prop_params.mu;
    sigma = prop_params.sigma;
    p = gauss_pdf(X, mu, sigma);
elseif (strcmp(type,'uniform'))
    a = prop_params.a;
    b = prop_params.b;    
    p = unifpdf(X,a,b);
end

end

function sample = sample_proposal(X, prop_params)

type = prop_params.type;

if (strcmp(type,'gauss'))
    %x_new ~ N(x_curr, Sigma); %proposal centered at x_curr
    Sigma = prop_params.sigma;
    sample = mvnrnd(X,Sigma); % x' ~ q(x'|x)
elseif (strcmp(type,'uniform'))
    a = prop_params.a;         %indep. proposal
    b = prop_params.b;
    sample = (b-a)*rand + a;
else
    %q(x|x')~N(x;x_map, H^-1)       %indep, centered at x_map
    %q(x|x')~N(x;x_curr, s^2 H^-1)  %adjustable scaling factor
                                    %to improve mixing
end

end

function p = gauss_pdf(X, mu, Sigma)
% Multivariate Gaussian distribution, pdf

d = size(Sigma, 2);
X  = reshape(X, [], d);  % make sure X is n-by-d and not d-by-n
X = bsxfun(@minus, X, mu(:)');
logp = -0.5*sum((X/(Sigma)).*X, 2); 
logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma);
logp = logp - logZ;
p = exp(logp);        

end

function [s] = sigmoid(a)
s = 1./(1+exp(-a));
end

function [w] = irls_lr(X,y,lambda,w)
% IRLS algorithm for L2-Penalized Logistic Regression

[n,p] = size(X);

if nargin < 3
    lambda = 0;
end

if nargin < 4
    w = zeros(p,1);
end

if lambda == 0
    % Use a weak prior to make numerically stable
    % (could use lambda=0 if you added a stabilization method)
    lambda = 1e-4;
end

if isscalar(lambda)
    vInv = 2*lambda*eye(p);
else
    vInv = 2*lambda;
end

max_num_iter = 100;
for i = 1:max_num_iter
    w_old = w;
    Xw = X*w;
    yXw = y.*Xw;                 %y in {-1,+1}
    sig = 1./(1+exp(-yXw));      %p(y|x,w)
    Delta = sig.*(1-sig) + eps;  %weight matrix Sk
    z = Xw + (1-sig).*y./Delta;  %target response
    %w = (X'*diag(sparse(Delta))*X + 2*lambda*eye(p))\X'*diag(sparse(Delta))*z;
    Xd = X'*diag(sparse(Delta));
    R = chol(Xd*X + vInv);
    w = R\(R'\Xd*z);
    
    %fprintf('iter = %d, f = %.6f\n',i,sum(abs(w-w_old)));
    if sum(abs(w-w_old)) < 1e-9
        %fprintf('L2-LogReg: Done\n');
        break;
    end
end

end
