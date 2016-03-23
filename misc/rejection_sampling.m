function [] = rejection_sampling()
%% rejection sampling
clear all; close all;
rng('default');

%% target and approx distribution

x=linspace(0,10,1e3);

%target distribution
k=1.65; %dof parameter
p = @(x) x.^(k-1).*exp(-x.^2/2);  %x>=0

%proposal distribution
mu = 0; Sigma = 3;
q = gauss_pdf(x, mu, Sigma);

%scaling constant
M = max(p(x)./q');

figure;
plot(x,p(x)); hold on;
plot(x,M*q); legend('target','proposal'); 
xlabel('x'); ylabel('density');

%% rejection sampling
n=1e3; nr=0;
z = zeros(n,1);
i = 1;
while i < n    
    xi = mvnrnd(mu, Sigma);
    u = rand(1);       
    if (u < p(xi)/(M*gauss_pdf(xi, mu, Sigma)))
        %accept
        z(i) = xi;
        i=i+1;        
    else
        %reject
        nr=nr+1;
    end        
end

fprintf('mean: %.2f\n', mean(z));
fprintf('std: %.4f\n', std(z));
fprintf('percent accepted: %.4f\n', n/(nr+n));

%% generate plots
figure;
hist(z);
title('histogram of target distribution'); 
ylabel('p(x)'); xlabel('x');

end

%% Multivariate Gaussian distribution, pdf
function p = gauss_pdf(X, mu, Sigma)
% X(i,:) is i'th case

d = size(Sigma, 2);
X  = reshape(X, [], d);  % make sure X is n-by-d and not d-by-n
X = bsxfun(@minus, X, mu(:)');
logp = -0.5*sum((X/(Sigma)).*X, 2); 
logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma);
logp = logp - logZ;
p = exp(logp);        

end

%% notes
