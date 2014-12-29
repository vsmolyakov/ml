%% Comparison of EM and variational inference algorithms 
%  for Latent Dirichlet Allocation (LDA) topic model
%
%  References:
%  D. Blei, A. Ng, and M. Jordan, "Latent Dirichlet Allocation", JMLR 2003
%  D. Blei and J. Lafferty, "Topic Models", 2009
%  T. Minka, "Estimating a Dirichlet Distribution", 2012

%% synthetic dataset
[A]=generate_data();

%% LDA
K=4; %fix the number of topics
[em_params,var_params]=LDA(A,K);

%% generate plots

%plot word distributions for each of the K topics using {em/var}_params.beta
figure;
for topic=1:K
    subplot(2,K,topic);
    plot(em_params.beta(:,topic)); hold on; axis tight;
    if (topic==1),title('EM Algorithm (LDA)'); xlabel('words'); ylabel('\beta_{w|z}');end
    legend_str=['topic ',num2str(topic)]; legend(legend_str,'Location','northeast');
    
    subplot(2,K,K+topic);
    plot(var_params.beta(:,topic)); hold on;  axis tight;
    if (topic==1),title('Variational Algorithm (LDA)'); xlabel('words'); ylabel('\beta_{w|z}');end
    legend_str=['topic ',num2str(topic)]; legend(legend_str,'Location','northeast');    
end

%compute topic similarity between EM and variational algorithms
V=size(A,1);
chi2_kernel=zeros(K,K); hist_intersection=zeros(K,K); bhattacharyya_coef=zeros(K,K); kl_kernel=zeros(K,K);
for i=1:K
    for j=1:K
        chi2_sum=0; hist_sum=0; bhat_sum=0; kldi_sum=0; %reset sums
        for v=1:V
            chi2_sum=chi2_sum+(em_params.beta(v,i)-var_params.beta(v,j)).^2/(em_params.beta(v,i)+var_params.beta(v,j));
            hist_sum=hist_sum+min(em_params.beta(v,i),var_params.beta(v,j));
            bhat_sum=bhat_sum+sqrt(em_params.beta(v,i)*var_params.beta(v,j));
            kldi_sum=kldi_sum+em_params.beta(v,i)*log((em_params.beta(v,i)+eps)/(var_params.beta(v,j)+eps));
        end
        chi2_kernel(i,j)=exp(-chi2_sum); hist_intersection(i,j)=hist_sum; bhattacharyya_coef(i,j)=bhat_sum; kl_kernel(i,j)=exp(-kldi_sum);        
    end
end

%Hungarian Algorithm to find min-cost topic matching
%between EM and Variational Algorithms in O(K^3) time
[zk_chi2,cost_chi2] = munkres(ones(K,K)-chi2_kernel);
[zk_bc,  cost_bc]   = munkres(ones(K,K)-bhattacharyya_coef);
[zk_hist,cost_hist] = munkres(ones(K,K)-hist_intersection);
[zk_kl,  cost_kl]   = munkres(ones(K,K)-kl_kernel);

cost_em_var=[cost_chi2,cost_bc,cost_hist,cost_kl];
zk_em_var=[zk_chi2',zk_bc',zk_hist',zk_kl'];

%[idx]=find(abs(cost-min(cost))<eps);
fprintf('max matching assignment em_var:\n');
mode(zk_em_var,2)

figure;
bar(cost_em_var,'LineWidth',2.0);
title('Topic Matching Between EM and Variational Algorithms');
ylabel('Topic Matching Error (Hungarian Algorithm)');
set(gca,'XTick',1:4); xlim([0 5]); legend('EM-Var');
set(gca,'XTickLabel',{'chi^2 kernel','bhattacharyya coef','hist intersection','KL kernel'});
