function [em_params, var_params]=LDA(A,K)
%% set LDA model parameters

%number of documents
D=size(A,2);

%dictionary size
nd_max=size(A,1);

%initialize word distribution beta
eta=ones(nd_max,1);   %uniform dirichlet prior on words
beta=zeros(nd_max,K);
for k=1:K
    beta(:,k)=dirrnd(eta);
    beta(:,k)=(beta(:,k)+eps); %to avoid beta(:,k)=0
    beta(:,k)=beta(:,k)/sum(beta(:,k)); %renormalize
end

% initialize each topic as the average of the overall word frequency
% and the frequency in a randomly chosen document
%[~,ind]=sort(rand(D,1)); 
%for k=1:K
%    rnd_doc=ind(k); 
%    nw=word_count(news.data{rnd_doc},nd_max)'; 
%    nw=nw/sum(nw); 
%    beta(:,k) = 0.5*Nw + 0.5*nw; %Nw: overal, nw: randomly chosen
%end

%initialize topic proportions theta and cluster assignments z
alpha=ones(K,1);   %uniform dirichlet prior on topics
z=zeros(nd_max,D); %cluster assignments (z_{i,d})
for d=1:D
    theta=dirrnd(alpha);
    wdn_idx=find(A(:,d));
    for i=1:length(wdn_idx)
       [~,z_idx]=max(mnrnd(1,theta)); %take most probable assignment        
       z(i,d)=z_idx;
    end
end

em_params.K = K;             % number of topics
em_params.D = D;             % number of documents
em_params.nd_max=nd_max;     % max number of words across all docs
em_params.alpha = alpha;     % dirichlet prior counts for topics
em_params.beta = beta;       % each column is beta_wi|zi
em_params.z=z;               % topic assignments

%init variaitonal parameters
gamma=ones(D,K);                    %symmetric dirichlet       
lambda=ones(K,nd_max)/nd_max;       %symmetric dirichlet
phi=ones(D,nd_max,K);               %equiprobable topics

for topic=1:K
    phi(:,:,topic)=randi(K,D,nd_max);
end

% initialize each topic as the average of the overall word frequency
% and the frequency in a randomly chosen document
%[~,ind]=sort(rand(D,1)); 
%for k=1:K
%    rnd_doc=ind(k); 
%    nw=word_count(news.data{rnd_doc},nd_max)'; 
%    nw=nw/sum(nw); 
%    beta_var(k,:) = 0.5*Nw + 0.5*nw; %Nw: overal, nw: randomly chosen
%end

var_params.K = K;             % number of topics
var_params.D = D;             % number of documents
var_params.nd_max=nd_max;     % max number of words across all docs
var_params.lambda=lambda;     % word frequencies free variational parameter
var_params.gamma=gamma;       % topic proportions free variational parameter
var_params.phi=phi;           % assignments free varational parameter
var_params.alpha=alpha;       % original topic proportions
var_params.beta=beta;         % original word frequencies
var_params.z=z;               % topic assignments

%% EM inference
if(1)
fprintf('EM inference algorithm...\n');
em_iter=10;
log_likelihood_em=zeros(em_iter,1);
log_likelihood_em_delta=zeros(em_iter,1);
for i=1:em_iter
    %compute expected value of log joint wrt to posterior
    %expressed in terms of document statistics (counts)    
    [nz,nwz,nlogtheta]=Estep(em_params,A);
    
    %update word distribution beta and mixing parameter alpha
    %based on the counts found in the Estep
    [em_params,loglik_mstep,loglik_delta]=Mstep(em_params,nz,nwz,nlogtheta);       

    log_likelihood_em(i)=loglik_mstep;
    log_likelihood_em_delta(i)=loglik_delta;
end

figure;
plot(log_likelihood_em);
title('LDA Topic Inference via EM Algorithm');
xlabel('EM iterations'); ylabel('log likelihood');

figure;
plot(log_likelihood_em_delta);
title('Delta Log Joint M-step');
xlabel('EM iterations'); ylabel('J_{new}-J_{old}');

log_likelihood_em
log_likelihood_em_delta
end

%% Variational inference
%min KL(q||p) = max ELBO
%q(theta,z,beta)=P_k q(beta_k|lambda_k) P_d q(theta_d|gamma_d) P_i q(z_id|phi_id)

if(1)
fprintf('variational inference algorithm...\n');
var_iter=10;
log_likelihood_var=zeros(var_iter,1);
log_likelihood_var_delta=zeros(var_iter,1);
for i=1:var_iter    
    
    %objective before update (see Blei 2003)
    [Jold]=elbo_objective(var_params);
    %fprintf('Jold: %f\n',Jold);    
        
    %update variational parameters
    [var_params]=mean_field_update(var_params,A);   
            
    %objective after update (see Blei 2003)
    [Jnew]=elbo_objective(var_params);   
    %fprintf('Jnew: %f\n',Jnew);
    
    loglik_mf=Jnew;
    loglik_delta_mf=Jnew-Jold;
    
    log_likelihood_var(i)=Jold;
    log_likelihood_var_delta(i)=loglik_delta_mf;    
    
    fprintf('Jnew-Jold: %f\n',Jnew-Jold);    
    %if (abs(loglik_delta_mf) < 10)
    %    break;
    %end    
end

%update alpha and beta
for topic=1:K
    var_params.alpha(topic,:)=sum(var_params.gamma(:,topic));
    var_params.beta(:,topic)=var_params.lambda(topic,:)/sum(var_params.lambda(topic,:));
end    

%update topic assignments
for d=1:D
    wdn_idx=find(A(:,d));
    for i=1:length(wdn_idx)
       [~,z_idx]=max(var_params.phi(d,wdn_idx(i),:)); %take most probable assignment        
       var_params.z(i,d)=z_idx;
    end
end

figure;
plot(log_likelihood_var);
title('LDA Topic Inference via Mean-Field Approximation');
xlabel('Mean-Field Iterations'); ylabel('log likelihood lower bound');

figure;
plot(log_likelihood_var_delta);
title('Delta Log Joint M-step');
xlabel('Mean-Field Iterations'); ylabel('J_{new}-J_{old}');

log_likelihood_var
log_likelihood_var_delta
end

figure;
semilogy(log_likelihood_em,'-b'); hold on; 
semilogy(log_likelihood_var,'--r');
title('LDA Topic Inference'); legend('EM','Variational ELBO');
xlabel('iterations'); ylabel('log likelihood');


function [nz, nwz, nlogtheta]=Estep(em_params,A)

%top-level counts
nz=zeros(em_params.K,1);
nwz=zeros(em_params.nd_max,em_params.K);
nlogtheta=zeros(em_params.K,1);

num_docs=size(A,2);
for d=1:num_docs
    
    doc=A(:,d);
    wdn_idx=find(doc);
    
    %zero document-level counts
    nz_d=zeros(em_params.K,1);                  
    nwz_d=zeros(em_params.nd_max,em_params.K);    
    
    %sample assignments from the posterior
    %P(z_i|z^-i,w) ~ beta_wi|z_i P(z_i|z^-i)
    %P(z_i|z^-i)   ~ alpha(z_i) + n^-i(zi)
    [z]=lda_gibbs(em_params,wdn_idx);
    em_params.z(:,d)=z;
    
    %update counts based on new assignments
    for i=1:length(wdn_idx)
        nz_d(z(i))=nz_d(z(i))+1;
        nwz_d(wdn_idx(i),z(i))=nwz_d(wdn_idx(i),z(i))+1;
    end

    %update top-level statistic
    nz=nz+nz_d;
    nwz=nwz+nwz_d;
    
    %E[log(theta_z)]=E[log(num/den)]=E[log(num)]-E[log(den)]]=
    %psi(alpha_zi+n^-i(z_i))-psi(alpha+n-1)
    nlogtheta=nlogtheta+psi(em_params.alpha+nz)-psi(sum(em_params.alpha+nz)-1);  
end

function [em_params,log_likelihood,loglik_delta]=Mstep(em_params,nz,nwz,nlogtheta)

K=em_params.K;
D=em_params.D;
alpha=em_params.alpha;
beta=em_params.beta;

%Note: nz, nwz, ntheta are counts over all D documents
%objective before the update
log_joint_before=-D*sum(gammaln(alpha))+D*gammaln(sum(alpha))+dot(alpha,nlogtheta);
%non-zero elements for nd_max size matrix
nwz_idx=find(nwz(:)); %beta_idx=find(beta(:)); notzero_idx=intersect(nwz_idx,beta_idx);
log_joint_before=log_joint_before+dot(nwz(nwz_idx),log(beta(nwz_idx)));

for k=1:K
    beta(:,k)=nwz(:,k)/nz(k);
    %alpha(k)=alpha(k)+nz(k);
end

%use newton updates for alpha:
%see "Estimating a Dirichlet Distribution" by T. Minka
%update alpha in the direction of stepest increase of the log joint obj
newton_iter=5;
for it = 1:newton_iter
    g = -D*psi(alpha) + D*psi(sum(alpha)) + nlogtheta; % gradient w.r.t alpha
    q = -D*psi(1,alpha);  %psi(1,alpha) - 1st deriv of digamma
    qa= D*psi(1,sum(alpha)); %see MLE estimation of dirichlet parameters
    b = sum(g./q)/(1/qa + sum(1./q));
    alpha = alpha - (g-b)./q;    %update alpha (constant step-size)
    alpha(alpha < 0) = 1e-3/K;   %make sure alpha > 0 (feas constr)
end;

%update em_params
em_params.alpha=alpha;
em_params.beta=beta;

%objective after update
log_joint_after=-D*sum(gammaln(alpha))+D*gammaln(sum(alpha))+dot(alpha,nlogtheta);
%non-zero elements for nd_max size matrix
nwz_idx=find(nwz(:)); %beta_idx=find(beta(:)); notzero_idx=intersect(nwz_idx,beta_idx);
log_joint_after=log_joint_after+dot(nwz(nwz_idx),log(beta(nwz_idx)));

fprintf('delta objective=%f\n',log_joint_after-log_joint_before);
log_likelihood=log_joint_after;
loglik_delta=log_joint_after-log_joint_before;


function [z]=lda_gibbs(em_params, doc)

alpha=em_params.alpha;
beta=em_params.beta;
K=em_params.K;
nd_max=em_params.nd_max;

%init to equiprobable assignments
p_cond=ones(K,1)/K;
p_gibbs=ones(nd_max,K)./repmat(K,nd_max,K);

%init
z=zeros(nd_max,1);
nz=zeros(K,1);

%introduce beta_doc for each document:
%1. to avoid division by zero (beta_doc size is sized for each doc)
%2. to extract word indices for each doc (collection of word idx):
%   beta(doc(word)) is correct, beta(word) is wrong
beta_doc=zeros(length(doc),K);
for word=1:length(doc), beta_doc(word,:)=beta(doc(word),:); end

ngibbs=1e1;
for i=1:ngibbs
    [~,ind] = sort(rand(1,length(doc))); % randomize order of updates
    for ii=1:length(doc)
        word=ind(ii);
        %n^-i(z_i): subtract one to exclude ith element from a non-zero count
        if (z(word)>0), nz(z(word)) = nz(z(word))-1; end

        if (z(word)>0)  %zi \in {1,...,K}
            p_cond(z(word))=(alpha(z(word))+nz(z(word)))/(sum(alpha)+length(doc)-1);
            if (sum(p_cond)>0), p_cond=p_cond/sum(p_cond); end
            p_gibbs(word, z(word))=(beta_doc(word,z(word))*p_cond(z(word)))/(dot(beta_doc(word,:),p_cond));
            if (sum(p_gibbs(word,:))>0), p_gibbs(word,:)=p_gibbs(word,:)/sum(p_gibbs(word,:)); end
            %if (isnan(sum(p_gibbs(word,:))) || sum(p_gibbs(word,:))==0)
            %    p_gibbs(word,:)=ones(1,K)/K; %init to [1/K,...,1/K] if NaN (division by zero)
            %end
        else
            %init (i==1)
            p_cond(:)=1/(sum(alpha)+length(doc)-1);
            if (sum(p_cond)>0), p_cond=p_cond/sum(p_cond); end
            p_gibbs(word,:)=(beta_doc(word,:)*p_cond(1))/(dot(beta_doc(word,:),p_cond)); 
            if (sum(p_gibbs(word,:))>0), p_gibbs(word,:)=p_gibbs(word,:)/sum(p_gibbs(word,:)); end
            %if (isnan(sum(p_gibbs(word,:))) || sum(p_gibbs(word,:))==0)
            %    p_gibbs(word,:)=ones(1,K)/K; %init to [1/K,...,1/K] if NaN (division by zero)
            %end            
        end
    
    %if (sum(p_gibbs)-1 > 1e-6), fprintf('sum(p_gibbs)=%d\n',sum(p_gibbs)); p_gibbs=p_gibbs/sum(p_gibbs); end    
    %if (sum(sum(isnan(p_gibbs))) > 0), fprintf('p_gibbs: %d\n',p_gibbs(word,:)); end
    
    %sample z_i from p_gibbs
    cdf=cumsum(p_gibbs(word,:)); u=rand; idx=min(find(cdf>=u));
    %z(word)=p_gibbs(word,:); %soft assignment    
    z(word)=idx; %hard assignment
    
    %update assignment
    nz(z(word))=nz(z(word))+1;
    end
end

function [var_params] = mean_field_update(var_params,A)

K=var_params.K;         
D=var_params.D;         
nd_max=var_params.nd_max;

lambda=var_params.lambda;      % word frequencies variational parameter
gamma=var_params.gamma;        % topic proportions variational parameter
phi=var_params.phi;            % assignments varational parameter

eta=ones(nd_max,1);            % dirichlet word prior

%gamma=ones(D,K)/K;             % check init         
%lambda=zeros(K,nd_max);        % check init
%phi=zeros(D,nd_max,K);         % check init

alpha_var=var_params.alpha;     % fixed to 1 (prior)
beta_var=var_params.beta;       % fixed to 1/nd_max (prior)

ndw=zeros(D,nd_max);           %word counts for each document
for d=1:D    
    %q(theta,z,beta)=
    %P_k q(beta_k|lambda_k) P_d q(theta_d|gamma_d) P_i q(z_id|phi_id)    
    doc=A(:,d);
    wdn_idx=find(doc);
    
    ndw(d,:)=word_count(wdn_idx,nd_max);

    %update gamma_d
    for topic=1:K
        gamma(d,topic)=alpha_var(topic)+dot(ndw(d,:),phi(d,:,topic));
    end
    
    %update phi_d,n
    for word=1:length(wdn_idx)
        phi(d,word,:)=exp( psi(gamma(d,:)')+psi(lambda(:,word))-psi(sum(lambda,2)) );
        if (sum(phi(d,word,:))>0) %to avoid 0/0 NaN
            phi(d,word,:)=phi(d,word,:)/sum(phi(d,word,:)); %normalize phi
        end
    end
end

%update lambda given ndw for all documents
for topic=1:K
    lambda(topic,:)=eta'+dot(ndw,phi(:,:,topic));   %check dot on matrices!
end

var_params.lambda=lambda;     
var_params.gamma=gamma;      
var_params.phi=phi;


function [nw] = word_count(d,m)
%d: data, m: vocab size, d(i):word index
nw = zeros(1,m); 
for i=1:length(d), 
    nw(d(i)) = nw(d(i))+1; 
end

function [J]=elbo_objective(var_params)
%see Blei 2003

J_t1=gammaln(sum(var_params.alpha))-sum(gammaln(var_params.alpha))+...
     dot((var_params.alpha-1),psi(sum(var_params.gamma,1)-psi(sum(sum(var_params.gamma)))));

J_t2=dot(sum(sum(var_params.phi,2),3), psi(sum(var_params.gamma,2))-psi(sum(sum(var_params.gamma))));    

beta_idx=find(var_params.beta(:)); phi_kn=sum(var_params.phi,3)';    
J_t3=dot(var_params.beta(beta_idx),phi_kn(beta_idx));

J_t4=-gammaln(sum(sum(var_params.gamma)))+sum(gammaln(sum(var_params.gamma)))+...
      dot((sum(var_params.gamma,1)-1),psi(sum(var_params.gamma,1)-psi(sum(sum(var_params.gamma)))));

phi_idx=find(var_params.phi);
J_t5=-dot(var_params.phi(phi_idx),log(var_params.phi(phi_idx)));

J=J_t1+J_t2+J_t3+J_t4+J_t5;


function [w]=lda_model(em_params,var_params,A)
%% Generative Model
%  can be used to generate word counts given inferred parameters
%  and compare resulting word counts to actual document
%  (see algebra on words by google)

%topic distributions
eta=ones(W,1);   %uniform prior
beta=zeros(W,K);
for kk=1:K
    beta(:,kk)=dirrnd(eta);
end

%word frequencies
alpha=ones(K,1);  %uniform dirichlet prior
theta=zeros(K,D); %mixing proportions
z=zeros(K,D);     %assignments
w=zeros(W,D);     %word indices
%wcat=zeros(W,D);  %words

for dd=1:D
    theta(:,dd)=dirrnd(alpha); 
    wdn_idx=find(A(:,dd));
    for i=1:length(wdn_idx)
       z(:,dd)=mnrnd(1,theta(:,dd));   %sample assignments
       [z_max,z_idx]=max(z(:,dd));     %most probable assignment
       
       w_temp=mnrnd(1,beta(:,z_idx));  %sample a word
       [w_max,w_idx]=max(w_temp);      %most probable word
       w(i,dd)=w_idx;                  %word index
       %wcat(i,dd)=A(w_idx,dd);        %word categorical    
    end
end
