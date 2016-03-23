%% sgd_lr_script

clear all; close all;
options.display_plots = 1;

%% grid parameter search

batchsize = logspace(1,3,3);  %[10, 100, 1000];
lambda = logspace(-4,-2,2);   %[1e-4, 1e-2];
method = [0,1];               %0:sgd, 1:sgld
kappa = [0.8,1];              %(0.5,1]

[X1,X2,X3,X4] = ndgrid(batchsize, lambda, kappa, method);
num_exp = numel(X1);

J_rec = {}; t_rec = {}; err_rec = {};
for i=1:num_exp
    [~, J_rec{i}, t_rec{i}, err_rec{i}] = sgld_lr(X1(i), X2(i), X3(i), X4(i));
    
    %compute performance statistic
    cv_err(i) = err_rec{i}(end); 
end

SAVE_PATH = './';
save([SAVE_PATH,'sgd_lr.mat']);

%% generate plots
% heatmap for best performance and individual 2d plots for sensitivity
if (options.display_plots)

SAVE_PATH = './';
load([SAVE_PATH,'sgd_lr.mat']);

%find min error parameters
[~,min_idx] = min(cv_err);
[x1_opt,x2_opt,x3_opt,x4_opt] = ind2sub(size(X1),min_idx);  %i_opt = x1_opt, j = x2, k = x3

J_temp = reshape(J_rec,size(X1));
t_temp = reshape(t_rec,size(X1));
err_temp = reshape(err_rec,size(X1));

%norm(theta) vs lambda
figure; legend_str = cell(length(lambda),1);
for i=1:length(lambda)
    A = t_temp(x1_opt,i,x3_opt,x4_opt);
    plot(reshape(A{:},numel(A{:}),1)); hold on;
    ylabel('norm theta'); xlabel('num updates');
    legend_str{i} = strcat('lambda = ', num2str(lambda(i)));
end
legend(legend_str);

%J vs batchsize
figure; legend_str = cell(length(batchsize),1);
for i=1:length(batchsize)
    A = J_temp(i,x2_opt,x3_opt,x4_opt);
    semilogy(reshape(A{:},numel(A{:}),1)); hold on;
    ylabel('cost'); xlabel('num updates');
    legend_str{i} = strcat('batch = ', num2str(batchsize(i)));
end
legend(legend_str);

%J vs method
figure; legend_str = cell(length(method),1);
for i=1:length(method)
    A = J_temp(x1_opt,x2_opt,x3_opt,i);
    semilogy(reshape(A{:},numel(A{:}),1)); hold on;
    ylabel('cost'); xlabel('num updates');
end
legend('SGD', 'SGLD');

%J vs kappa
figure; legend_str = cell(length(kappa),1);
for i=1:length(kappa)
    A = J_temp(x1_opt,x2_opt,i,x4_opt);
    plot(reshape(A{:},numel(A{:}),1)); hold on;
    ylabel('cost'); xlabel('num updates');
    legend_str{i} = strcat('kappa = ', num2str(kappa(i)));
end
legend(legend_str);

%err_rec vs method
figure; legend_str = cell(length(method),1);
for i=1:length(method)
    A = err_temp(x1_opt,x2_opt,x3_opt,i);
    semilogy(cummean(reshape(A{:},numel(A{:}),1))); hold on;
    ylabel('classification error'); xlabel('num updates');
end
legend('SGD', 'SGLD');

%heatmap
%{
temp = reshape(cv_err,size(X1));
figure;
subplot(121); colormap('hot');
imagesc(temp(:,:,1)); title('classification error'); colorbar;
subplot(122); colormap('hot');
imagesc(temp(:,:,2)); 
colorbar;
%}

end

%% notes
