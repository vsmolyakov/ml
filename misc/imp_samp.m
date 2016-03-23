%% Importance Sampling Example
%
%  E[f(x)] = int_x f(x)p(x)dx = int_x f(x)[p(x)/q(x)]q(x) dx
%  = sum_i f(x_i)w(x_i), where x_i ~ q(x)
%  e.g. for f(x) = 1(x \in A), E[f(x)] = P(A)

clear all; close all;

%% f(x),  x >= 0
f = @(x) 2*sin((pi/1.5)*x);

%% target distribution p(x), x >= 0 (assume unnormalized)
k=1.65; %dof parameter
p = @(x) x.^(k-1).*exp(-x.^2/2);

%% proposal distribution q(x) (assume unnormalized)
mu = 0.8; sigma = sqrt(1.5); %est variance is sensitive to q(x)~N(mu,sigma)
c=2; %fix c, s.t. p(x) < c q(x), alternatively min D_alpha(p||q)
q = @(x) c * 1/sqrt(2*pi*1.5) * exp(-(x-mu).^2/(2*sigma^2));

%% importance sampling

num_iter = [1e1,1e2,1e3,1e4,1e5,1e6]; num_trials = 10;
f_est = {{}}; w_var = {{}};

for trial = 1:num_trials
    for iter = 1:length(num_iter)
        %rng(iter);
        fprintf('trial: %d, num_iter: %d\n', trial, num_iter(iter));
    
        %sample from proposal
        x = normrnd(mu,sigma,num_iter(iter),1);
  
        %discard x<0 samples
        x_pos = x.*(x>=0); idx = find(x_pos);
    
        %compute weights
        w = p(x_pos(idx))./q(x_pos(idx));
    
        fw = (w/sum(w)).*f(x_pos(idx));
        
        f_est{iter}{trial} = sum(fw);
        w_var{iter}{trial} = var(w/sum(w));
    end
end

%average E[f(x)] estimates
f_est_mean = zeros(size(f_est,2),1);
for i=1:size(f_est,2)
    f_est_mean(i) = mean([f_est{i}{:}]);
end

%average importance weight variance
w_var_mean = zeros(size(w_var,2),1);
for i=1:size(w_var,2)
    w_var_mean(i) = mean([w_var{i}{:}]);
end

%ground truth
fp = @(x) 2*sin((pi/1.5)*x).*(x.^(k-1).*exp(-x.^2/2));
f_gt = integral(fp,0,5);

%MSE
f_diff = f_est_mean - f_gt *ones(size(f_est_mean));
MSE = f_diff.*f_diff;

%effective number of samples
neff = num_iter'./(1+w_var_mean);

%% generate plots

figure; 
xx=linspace(0,8,100);
plot(xx, p(xx),'-r','linewidth',2.0); hold on; plot(xx, q(xx),'-b','linewidth',2.0); hold on;
plot(xx, p(xx)./q(xx),'--k','linewidth',2.0); hold on; plot(xx, p(xx).*f(xx),'-g','linewidth',2.0); hold on; grid on;
legend('p(x) target distribution','q(x) proposal distribution','p(x)/q(x) importance weight','p(x)f(x) integrand');
title('importance sampling'); xlabel('x'); ylabel('f(x)');

figure;
semilogx(num_iter, f_est_mean,'-or'); hold on;
semilogx(num_iter, f_gt*ones(size(num_iter)),'--ok'); hold on;
legend('E[f(x)] estimate', 'ground truth E[f(x)]','location','SE'); xlabel('number of iterations'); ylabel('E[f(x)] estimate');
title('importance sampling estimate of E[f(x)] vs iterations');

figure;
loglog(num_iter, w_var_mean,'-ob'); hold on;
loglog(num_iter, 1./sqrt(num_iter),'-or');
xlabel('number of samples'); ylabel('variance'); legend('avg w variance','1/sqrt(n)','location','SW');
title('E[f(x)] estimator variance vs number of samples');

figure;
semilogx(num_iter,neff./num_iter','-ob'); title('effective number of samples');
xlabel('number of samples'); ylabel('n effective / n'); legend('n_{eff} / n','location','SE');

