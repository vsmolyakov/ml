
function [theta, J_rec, t_rec, err_rec] = sgld_lr(batchsize, lambda, kappa, method)
%% Stochastic Gradient Langevin Dynamics for Bayesian Logistic Regression
%  with Laplace prior
% min NLL(w) = min -sum_{i=1}^{n} [y_i log mu_i + (1-y_i) log(1-mu_i)]
% where mu_i = sigm(w'x_i)

%clear all; close all; 
rng('default');

%% dataset
%dataset = 'a9a'; %'synthetic', 'mnist'
dataset = 'synthetic';

if (strcmp(dataset,'synthetic'))
% synthetic data
n=1e4;

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
elseif (strcmp(dataset,'a9a'))
% a9a dataset
%https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

DATA_PATH = './';
file_name_train = [DATA_PATH, 'a9a'];
%file_name_test = [DATA_PATH, 'a9a.t'];

[y, X] = libsvmread(file_name_train); 
%[y_train, X_train] = libsvmread(file_name_train);
%[y_test, X_test] = libsvmread(file_name_test);

elseif (strcmp(dataset,'mnist'))
    
class = [2 3]; %binary classification
%[Xtrain, ytrain, Xtest, ytest] = mnistLoad(class);
[X, y_uint, ~, ~] = mnistLoad(class);

y = zeros(size(X,1),1);
y_supp = [-1, 1];
for i=1:length(y_supp)
    if (length(class) ~= length(y_supp))
        fprintf('support mismatch'\n'); break;
    end
    idx = find(y_uint == class(i));
    y(idx) = y_supp(i);
end
clear y_uint;

else
    
fprintf('please choose a dataset\n');

end

%add bias
X = [ones(size(X,1),1) X];

%% SGLD for logistic regression

num_sweeps = 1e1;                   %number of data sweeps
num_train = floor(0.8*size(X,1));   %number of training examples
theta = randn(size(X,2),1);         %theta0 ~ N(0,1);
%theta = zeros(size(X,2),1);

%batchsize = 200;                    %mini-batch size
num_batches = ceil(num_train / batchsize);

%step-size
alpha = 1e-2; %0.1;
tau0 = 10; %1e2
%kappa = 0.8; %(0.5,1]
eta_vec = zeros(num_batches*num_sweeps,1); 
for i=1:num_batches*num_sweeps
    eta_vec(i) = alpha/(tau0 + i)^(kappa);
end
fprintf('sum_t eta_t: %.4f\n', sum(eta_vec));
fprintf('sum_t eta_t^2: %.4f\n', sum(eta_vec.^2));
eta = reshape(eta_vec,num_batches,num_sweeps); 
%figure; plot(eta_vec); title('step size');
%xlabel('iterations'); ylabel('learning rate \eta_k');
% Leon Bottou, @(t) 1/(lambda*(t+t0)));
% Nic Schraudolph, @(t) eta0*t0/(t+t0));
clear eta_vec;

%l2 regularization
%lambda = 1e-5; %1e-9;

J_rec = zeros(num_batches,num_sweeps);
t_rec = zeros(num_batches,num_sweeps);
theta_rec = zeros(size(X,2),num_sweeps);
err_rec = zeros(num_batches,num_sweeps);

for sweep = 1:num_sweeps  

    rng(sweep);
    idx_train = randperm(num_train);

    X_train = X(idx_train,:); X_test = X(num_train+1:end,:);
    y_train = y(idx_train,:); y_test = y(num_train+1:end,:);
    
    %mini-batch size
    [batchdata, batchlabels] = make_batches(X_train, y_train, batchsize);
    if (num_batches ~= numel(batchlabels)), fprintf('mismatch in the number of batches\n'); end
        
    %SGD
    for batch = 1:num_batches    
       Xb = batchdata{batch}; yb = batchlabels{batch};
       [cost, grad_lik, grad_prior, ~] = sgld_lr_obj(theta, Xb, yb, lambda);       
       if (method) %SGLD
            theta = theta - (eta(batch,sweep) / 2) * (num_batches * grad_lik + grad_prior) + sqrt(eta(batch,sweep))*randn(size(X_train,2),1);
            %theta = theta - (eta(batch,sweep) / 2) * (grad_lik / size(Xb,1) + grad_prior) + sqrt(eta(batch,sweep))*randn(size(X_train,2),1);
       else %SGD
            theta = theta - (eta(batch,sweep) / 2) * (num_batches * grad_lik + grad_prior);
            %theta = theta - (eta(batch,sweep) / 2) * (grad_lik / size(Xb,1) + grad_prior);
       end
       
       J_rec(batch,sweep) = cost;
       t_rec(batch,sweep) = norm(theta,2);
       
       %compute accuracy
       y_pred = 2*(sigmoid(X_test*theta) >= 0.5)-1;
       y_err = length(find(y_pred-y_test))/length(y_test);    
       err_rec(batch,sweep) = y_err;
    end
    theta_rec(:,sweep) = theta;
    
    fprintf('dataset sweep %d, cost: %.4f\n',sweep,cost);
end

fprintf('norm theta: %.4f\n', norm(theta,2));
y_pred = 2*(sigmoid(X_test*theta) >= 0.5)-1;
y_err = length(find(y_pred-y_test))/length(y_test);
fprintf('classification error: %.4f\n', y_err);

%% generate plots

%{
figure;
plot(reshape(J_rec,numel(J_rec),1)); title('logistic regression');
xlabel('num updates'); ylabel('cost');

figure;
plot(reshape(t_rec,numel(t_rec),1)); title('logistic regression');
xlabel('num updates'); ylabel('theta l2 norm');

figure;
plot(cummean(reshape(err_rec,numel(err_rec),1)));
xlabel('num sweeps'); ylabel('classification error');        

if (strcmp(dataset,'synthetic'))
    figure;
    scatter(X(find(y==1),2),X(find(y==1),3), 'ro'); hold on;
    scatter(X(find(y==-1),2),X(find(y==-1),3), 'bo'); hold on;

    x2 = linspace(min(X(:,2))-1,max(X(:,2))+1,10);
    plot(x2, -(theta(2)/theta(3))*x2 - theta(1)/theta(3), '-r'); 
end
%}

end

%% lr objective with l2 penalty
function [cost, grad_lik, grad_prior, H] = sgld_lr_obj(theta, X, y, lambda)

n = length(y);
y01 = (y+1)/2;

%compute the objective
mu = sigmoid(X*theta);

mu = max(mu,eps);    %bound away from 0
mu = min(mu,1-eps);  %bound away from 1

cost = -(1/n)*sum(y01.*log(mu)+(1-y01).*log(1-mu)) + sum(lambda.*(theta(2:end).^2));

%compute gradient of the lr objective
grad_lik = X'*(mu-y01) + 2*lambda.*[0; theta(2:end)];
grad_prior = -sign(theta) + 2*lambda.*[0; theta(2:end)];

%compute hessian of the lr objective
H_lambda = 2*lambda*eye(length(theta)); H_lambda(1,1) = 0;
H = X'*diag(mu.*(1-mu))*X + H_lambda;

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
