function [theta] = sgd_lr()
%% SGD for logistic regreesion
% min NLL(w) = min -sum_{i=1}^{n} [y_i log mu_i + (1-y_i) log(1-mu_i)]
% where mu_i = sigm(w'x_i)

clear all; close all; rng('default');

%% synthetic data
n=1e3;

%mu1 = [2,2]; mu2 = [-2,-2];
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

%figure;
%scatter(X(:,1),X(:,2));

%% SGD for logistic regression

num_iter = 1e2;
theta = randn(size(X,2),1);
lambda = 1e-9; %regularization

tau0 = 10;
kappa = 1; %(0.5,1] 
eta = zeros(num_iter,1); 
for i=1:num_iter
    eta(i) = (tau0 + i)^(-kappa);
end
%figure; plot(eta)
%xlabel('iterations'); ylabel('learning rate \eta_k');
% Leon Bottou, @(t) 1/(lambda*(t+t0)));
% Nic Schraudolph, @(t) eta0*t0/(t+t0));

batchsize = 200;
[batchdata, batchlabels] = make_batches(X, y, batchsize);
num_batches = numel(batchlabels);

num_updates = 0;
J_rec = zeros(num_iter * num_batches,1);
t_rec = zeros(num_iter * num_batches,1);

for iter = 1:num_iter   
    
    for b = 1:num_batches
       Xb = batchdata{b}; yb = batchlabels{b};
       [J_cost, J_grad] = lr_objective(theta, Xb, yb, lambda);
       theta = theta - eta(iter) * (num_batches * J_grad);
       
       num_updates = num_updates + 1;
       J_rec(num_updates) = J_cost;
       t_rec(num_updates) = norm(theta,2);       
    end

    fprintf('iteration %d, cost: %.2f\n',iter,J_cost);
end

y_pred = 2*(sigmoid(X*theta) >= 0.5)-1;
y_err = length(find(y_pred-y))/n;
fprintf('classification error: %.4f\n', y_err);

%% generate plots

figure;
plot(J_rec); title('logistic regression');
xlabel('iterations'); ylabel('cost');

figure;
plot(t_rec); title('logistic regression');
xlabel('iterations'); ylabel('theta l2 norm');

figure;
scatter(X(find(y==1),1),X(find(y==1),2), 'ro'); hold on;
scatter(X(find(y==-1),1),X(find(y==-1),2), 'bo'); hold on;

x1 = linspace(min(X(:,1))-1,max(X(:,1))+1,10);
plot(x1, -(theta(1)/theta(2))*x1, '-r');


end

%% lr objective with l2 penalty
function [cost, grad, H] = lr_objective(theta, X, y, lambda)

n = length(y);
y01 = (y+1)/2;

%compute the objective
mu = sigmoid(X*theta);

mu = max(mu,eps);    %bound away from 0
mu = min(mu,1-eps);  %bound away from 1

cost = -(1/n)*sum(y01.*log(mu)+(1-y01).*log(1-mu)) + sum(lambda.*(theta.^2));

%compute gradient of the lr objective
grad = X'*(mu-y01) + 2*lambda.*theta;

%compute hessian of the lr objective
H = X'*diag(mu.*(1-mu))*X + 2*lambda*eye(length(theta));

end

%% sigmoid function
function [s] = sigmoid(a)
s = 1./(1+exp(-a));
end

%% mini-batch function
function [batchdata, batchlabels] = make_batches(X, y, batchsize)
  n = size(X,1);
  num_batches = ceil(n/batchsize);
  groups = repmat(1:num_batches,1,batchsize);
  groups = groups(1:n);
  batchdata = cell(1, num_batches);
  batchlabels = cell(1, num_batches);
  for i=1:num_batches
    batchdata{i} = X(groups == i,:);
    batchlabels{i} = y(groups == i,:);
  end
end

%% notes
