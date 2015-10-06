function [X,mu0,V0,alpha,pi,num_exp] = gen_data(n,d,K,type)

if (nargin < 1), n=1e2; end
if (nargin < 2), d=4;   end
if (nargin < 3), K=1e1; end
if (nargin < 4), type='gauss'; end

switch type
    case 'gauss'
        %ground truth mu and sigma
        mu0=randi(10,d,K); V0=zeros(d,d,K); %PD
        for k=1:K, A=randn(d,d); V0(:,:,k) = A'*A; end
        X=zeros(n,d,K);
        for kk=1:K, X(:,:,kk)=mvnrnd(mu0(:,k),V0(:,:,k),n); end
        %figure;
        %scatter(X(:,1),X(:,2)); hold on; grid on;
        %for k=1:K, plot2dgauss(mu0(:,k), V0(:,:,k)); hold on; end
    case 'mult'        
        %alternatively, using non-uniform alpha for posterior
        alpha = ones(d,1); num_exp = d; %1e2;
        X=zeros(n,d,K); pi=zeros(K,d);
        for iter = 1:K
            pi(iter,:) = dirrnd(alpha);
            X(:,:,iter) = mnrnd(num_exp,pi(iter,:),n); 
        end
        mu0=zeros(d,K); V0=zeros(d,d,K);
        %figure; image(X(:,:,1); colormap(gray(256));
end

