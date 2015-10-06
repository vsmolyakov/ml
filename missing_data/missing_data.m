%% missing data

clear all; close all;
set(0,'DefaultAxesFontSize',15,'DefaultAxesFontName','Helvetica','DefaultAxesFontAngle','Normal');
%TODO: compare different methods (mean or median vs parallel imp)
%TODO: to check: sum(x_sticks)>1

%% generate data (similar to GMM / topic models)
num_samples=2e2; d=12; K=0.5e1;
%[X,mu0,V0,~,~,num_exp] = gen_data(num_samples,d,K,'gauss');
[X,~,~,alpha,~,num_exp] = gen_data(num_samples,d,K,'mult');

%% introduce missing values
[m,n] = size(X(:,:,1)); percent_data = 0.2;
num_nan=round(percent_data*numel(X(:,:,1)));
num_blocks = 3; block_len = round(n/num_blocks);
X_obs = zeros(num_samples,d,K);
X_mis = zeros(num_samples,d,K); 
noise_type = 'block'; %{'block','random'}

switch noise_type
    case 'random'
        for iter=1:K    
            nan_vec = zeros(m*n,1);
            nan_idx = randsample(m*n,num_nan);
            nan_vec(nan_idx) = NaN;
            X_mis(:,:,iter) = reshape(nan_vec,m,n);
            X_obs(:,:,iter) = X(:,:,iter)+X_mis(:,:,iter);
        end
    case 'block'        
        %define block noise vectors
        noise_vec = zeros(num_blocks,d);
        noise_vec(1,1:block_len) = NaN;
        noise_vec(2,block_len+1:2*block_len) = NaN;
        noise_vec(3,2*block_len+1:3*block_len) = NaN;                         
        for iter=1:K
            %sample a block
            [~,block_idx] = find(mnrnd(1,ones(num_blocks,1)/num_blocks,num_samples));
            for kk = 1:num_blocks
                block_idx_k = find(block_idx == kk);
                X_mis(block_idx_k,:,iter) = repmat(noise_vec(kk,:),length(block_idx_k),1);
                X_obs(:,:,iter) = X(:,:,iter)+X_mis(:,:,iter);
            end
        end 
end

%remove rows with zero NaNs
%idx_row_nan = find(sum(isnan(X_obs),2));

%% parallel imputation (assumes we know num_samples)

alpha0 = sum(alpha); num_samples_nan = 1e1;
X_imp = zeros(num_samples,d,K); %imputed
for iter = 1:K
    for i=1:num_samples                                
        [idx_nan]=isnan(X_obs(i,:,iter)); num_nan = sum(idx_nan);
        if (~num_nan), continue; end %continue if no NaNs 
        alpha_nan = max(0, alpha(idx_nan)/(alpha0-num_nan)); %gets sparser (is this right?)
                        
        [stick_length] = stick_brk(X_obs(i,:,iter),alpha);
        %pi_nan = stick_length*dirrnd(alpha_nan);
        %pi_nan = pi_nan/sum(pi_nan); 

        X_imp(i,:,iter) = X_obs(i,:,iter); samples_nan = zeros(num_samples_nan, num_nan);
        for j=1:num_samples_nan, samples_nan(j,:) = stick_length*dirrnd(alpha_nan); end
        X_imp(i,idx_nan,iter) = round((num_exp-sum(X_obs(i,~idx_nan,iter)))*mean(samples_nan,1));  %check derivation
    end
end

%% compute error
X_err = zeros(num_samples,d,K);
imp_err_fro = zeros(K,1); imp_err_l1 = zeros(num_samples,iter);
for iter=1:K
    X_err(:,:,iter) = X(:,:,iter) - X_imp(:,:,iter);
    imp_err_l1(:,iter) = sum(abs(X_err(:,:,iter)),2) / num_samples;
    imp_err_fro(iter) = norm(X(:,:,iter) - X_imp(:,:,iter),'fro') / num_samples;
end

%% visualize results
figure; hist(imp_err_l1); xlabel('l1 imputation error'); ylabel('Histogram of Parallel Imputation Error for K trials');
figure; ecdf(mean(imp_err_l1,2)); xlabel('x: l1 imputation error'); title('Empirical CDF of Parallel Imputation (Avg of K trials)');
figure; boxplot(imp_err_l1); xlabel('iteration'); ylabel('l1 imputation error'); title('boxplot of l1 error vs iteration');
%figure; for iter=1:K, ecdf(imp_err_l1(:,iter)); hold on; end;
figure; imagesc(abs(X_err(:,:,iter))); colormap(gray(256));
figure; imagesc(imp_err_l1); colormap(gray(256));
