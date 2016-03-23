%% Gibbs Sampler for Ising model
%
%  p(xt=+1|x_{-t},y,theta) = sigm{2Jn_t - log(psi_t(-1)/psi_t(+1))}
%  where psi_t = N(y_t; x_t, sigma^2)

clear all; close all;

%% generate data
rng('default');
data = load('letterA.mat'); %128 x 128
img = double(data.A); %figure; imshow(data.A)
img_mean = mean(img(:));
img2 = +1*(img>img_mean) + -1*(img<img_mean); % x_i \in {+1,-1}
sigma  = 2; % noise level
y = img2 + sigma*randn(size(img2)); %y_i ~ N(x_i; sigma^2);

%% set model parameters
J = 1; % coupling strength (vary J \in [-10:10])

%% main loop

[rows,cols]=size(y); 
Px  = zeros(rows,cols); %P(x_i=1)
eta = zeros(rows,cols);  %sum_{j\in N(i)} x_i

%sigmoid function
sigm = @(u) 1./(ones(size(u))+exp(-u));

%neighbor mask
A=[0 1 0; 1 0 1; 0 1 0];

%init image
num_sweeps=20;
y_k = zeros(rows,cols,num_sweeps); y_k(:,:,1) = y;
for sweep = 1:num_sweeps-1
    rng(sweep);
    fprintf('image sweep: %d\n',sweep);   
    
    eta = conv2(y_k(:,:,sweep),A,'same');
    U = 2*J*eta - log(normcdf(-1,y_k(:,:,sweep),sigma)./normcdf(+1,y_k(:,:,sweep),sigma));
    Px = sigm(U);    
    y_k(:,:,sweep+1) = 2*binornd(1,Px)-ones(size(Px)); %y_k \in {+1,-1}
    
    if (mod(sweep,5)==0)
        figure; imshow(y_k(:,:,sweep)); str = sprintf('gibbs iter: %d', sweep); title(str);
    end
end

y_k_mean = mean(y_k,3);

figure; 
subplot(121); imshow(y); colorbar; title('original');
subplot(122); imshow(y_k_mean); colorbar; title('mean gibbs');


