function [samples] = mcmc_mh_gaussmix()
%% Metropolis-Hastings Algorithm for Sampling a Mixture of Gaussians

%clear all; close all;
rng('default');

%% define target (posterior) and proposal (approximating) distributions

%target distribution (gaussian mixture)
mu1 = [3]; mu2 = [-3]; d = length(mu1);
Sigma1 = eye(d); Sigma2 = eye(d);
%A1 = randn(d,d); Sigma1 = A1'*A1;
%A2 = randn(d,d); Sigma2 = A2'*A2;
pik = [0.3, 0.7];
target_params.mu =[mu1, mu2];
target_params.sigma = [1, 1];
target_params.weights = pik;

% proposal distribution
prop_params.type = 'gauss'; %'gauss'; %'uniform'
prop_params.mu = 0;   %N(mu,sigma)
prop_params.sigma = 10;

prop_params.a = -10;  %Unif[a,b]
prop_params.b = 10;

figure;
x = linspace(-10,10,1e3);
p = target(x,target_params);
plot(x,p,'-b'); hold on;
q = proposal(x, prop_params);
plot(x,q,'-r'); hold on;
legend('target','proposal');
xlabel('x');ylabel('p(x)');

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
plot(samples); title('trace plot');
xlabel('number of samples'); ylabel('samples');

figure;
nbins = 5*floor(max(samples)-min(samples));
[h,bins] = hist(samples,nbins);
bin_width = bins(2)-bins(1);
p_est = h/(sum(h)*bin_width);
bar(bins, p_est); hold on;
plot(bins, target(bins,target_params),'-r','linewidth',2);
xlabel('x'); ylabel('p(theta|x)'); 
legend('MCMC-MH','true posterior');

end

function p = target(X,target_params)
% P(theta|x) = sum_k pi(k) p(theta;mu_k,Sigma_k)

mus = target_params.mu;
sigmas = target_params.sigma;
weights = target_params.weights;

K = length(weights); n = size(X,1); p = zeros(n,1);
for k=1:K
    p = p + weights(k)*gauss_pdf(X,mus(k),sigmas(k));
end

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


%% notes

