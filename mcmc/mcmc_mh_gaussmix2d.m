function [samples] = mcmc_mh_gaussmix2d()
%% Metropolis-Hastings Algorithm for Sampling a Mixture of Gaussians in 2D

%clear all; close all;
rng('default');

%% define target (posterior) and proposal (approximating) distributions

%target distribution (gaussian mixture)
d = 2;  %4
K = 2;  %4
mu = zeros(d,K); 
mu(:,1) = [3, 0]'; mu(:,2) = [-3, 0]';
Sigma = zeros(d,d,K); 
A1 = randn(d,d); Sigma(:,:,1) = A1'*A1 + diag(randi(10,d,1));
A2 = randn(d,d); Sigma(:,:,2) = A2'*A2 + diag(randi(10,d,1));
pik = zeros(K,1); pik = [0.3, 0.7];
target_params.mu = mu;
target_params.sigma = Sigma;
target_params.weights = pik;

% proposal distribution
prop_params.type = 'gauss'; %'gauss'; %'uniform'
prop_params.mu = zeros(d,1);   %N(mu,sigma)
prop_params.sigma = 10*eye(d,d);

prop_params.a = -15;  %Unif[a,b]
prop_params.b = 15;

%visualize 2D target and proposal
[xg,yg] = meshgrid(-10:0.1:10, -10:0.1:10); 
[nxg,nxg]=size(xg);
X=[reshape(xg,nxg*nxg,1), reshape(yg,nxg*nxg,1)];
p = target(X,target_params);
q = proposal(X, prop_params);
figure; mesh(xg,yg,reshape(p,[nxg,nxg])); hold on;
%mesh(xg,yg,reshape(q,[nxg,nxg]));
%xlabel('x');ylabel('y');zlabel('p(x,y)')
%figure; [cc,h]=contour(xg,yg,reshape(p,[nxg,nxg]),30); hold on;
%[cc,h]=contour(xg,yg,reshape(q,[nxg,nxg]),30);

%% MH Algorithm
n = 5e3; %number of samples
num_chains = 3;
n_accept = zeros(num_chains,1);
s = zeros(n,d,num_chains);
burnin = floor(0.1*n);
alpha = zeros(n,1);

tic;
for chain = 1:num_chains

rng(chain);
s(1,:,chain) = sample_proposal(prop_params.mu,prop_params); %init

for i = 1:n
  x_curr = s(i,:,chain);
  x_new = sample_proposal(x_curr,prop_params);
  x_new = x_new(:)';
  
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

figure;
plot(s(:,:,1)); hold on;
plot(s(:,:,2)); hold on;
plot(s(:,:,3)); hold on;
title('trace plot');
xlabel('number of samples'); ylabel('samples');

[R,neff,Vh,W,B,tau,thin] = psrf(s(:,:,1),s(:,:,2),s(:,:,3));

%thin and merge samples after burnin
samples = [s(burnin+1:end,:,1);s(burnin+1:end,:,2);s(burnin+1:end,:,3)];
%samples = [s(burnin+1:thin:end,:,1);s(burnin+1:thin:end,:,2);s(burnin+1:thin:end,:,3)];


%% generate plots
figure;
plot(alpha); ylabel('alpha'); xlabel('num iter');

figure;
scatter(samples(:,1),samples(:,2)); title('trace plot'); hold on;
xlabel('x1'); ylabel('x2');
[xg,yg] = meshgrid(-15:0.1:15, -15:0.1:15); [nxg,nxg]=size(xg);
X=[reshape(xg,nxg*nxg,1), reshape(yg,nxg*nxg,1)];
p = target(X,target_params);
[cc,h]=contour(xg,yg,reshape(p,[nxg,nxg]),30);
xlim([-15,15]); ylim([-15,15]); legend('MCMC-MH','true posterior');

figure;
nbins = max(5*floor(max(samples)-min(samples)));
[h,bins] = hist3(samples,[nbins,nbins]);
bin_width = bins{1}(2)-bins{1}(1);
p_est = h./(sum(h(:))*bin_width);
bar3(bins{1}, p_est); hold on;

end

function p = target(X,target_params)
% P(theta|x) = sum_k pi(k) p(theta;mu_k,Sigma_k)

mus = target_params.mu;
sigmas = target_params.sigma;
weights = target_params.weights;

K = length(weights); n = size(X,1); p = zeros(n,1);
for k=1:K
    p = p + weights(k)*gauss_pdf(X,mus(:,k),sigmas(:,:,k));
end

end

function p = proposal(X, prop_params)
% Multivariate Gaussian distribution, pdf

type = prop_params.type;
d = size(X,2);

if (strcmp(type,'gauss'))
    mu = prop_params.mu;
    sigma = prop_params.sigma;
    p = gauss_pdf(X, mu, sigma);
elseif (strcmp(type,'uniform'))
    a = prop_params.a;
    b = prop_params.b;    
    if (d==1)
        p = unifpdf(X,a,b);
    else
        p = unifpdf(X(:,1),a,b);
    end
end

end

function sample = sample_proposal(X, prop_params)

type = prop_params.type;
d = size(X,2);

if (strcmp(type,'gauss'))
    %x_new ~ N(x_curr, Sigma); %proposal centered at x_curr
    Sigma = prop_params.sigma;
    sample = mvnrnd(X,Sigma); % x' ~ q(x'|x)    
elseif (strcmp(type,'uniform'))
    a = prop_params.a;         %indep. proposal
    b = prop_params.b;
    sample = (b-a)*rand(d,1) + a;
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


%% notes

