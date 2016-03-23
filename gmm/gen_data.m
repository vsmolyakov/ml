function [X,mu0,V0] = gen_data(n,d,K)

if (nargin < 1), n=1e3; end
if (nargin < 2), d=2;   end
if (nargin < 3), K=3;   end

%GMM generative model
alpha0=ones(K,1); pi=dirrnd(alpha0);

%ground truth mu and sigma
mu0=randi(20,d,K)-10*ones(d,K); V0=zeros(d,d,K); %PD
for k=1:K, A=randn(d,d); V0(:,:,k) = A'*A + 1e-1*eye(d); end

zi = mnrnd(1,pi,n); X=zeros(n,d);
for i=1:n
    k=find(zi(i,:));
    X(i,:)=mvnrnd(mu0(:,k),V0(:,:,k));
end

%figure;
%scatter(X(:,1),X(:,2)); hold on; grid on;
%for k=1:K, plot2dgauss(mu0(:,k), V0(:,:,k)); hold on; end
