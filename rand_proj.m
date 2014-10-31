function [P] = rand_proj(x, n_components)
%% random projection
%1. Gaussian: ~N(0,1/n_components)
%2. Sparse: -sqrt(s/n_components) w.p. 1/2s
%           0                     w.p. 1-1/s
%           +sqrt(s/n_components) w.p. 1/2s
%Input:
%x: data vector original space
%n_components: dimension of projected space
%Output:
%P: projection matrix

if (nargin < 2), n_components=1e1; end
orig_dim=length(x);

%Gaussian
P_gauss=(1/sqrt(n_components))*randn(n_components,orig_dim);
%imshow(P_gauss)

%Sparse
density=0.1; %rand %density \in (0,1)
s=1/density; pmf=[1/(2*s) 1-1/s 1/(2*s)];
MN_draw=mnrnd(1,pmf,n_components*orig_dim);
MN_keys=[-sqrt(s/n_components) 0 +sqrt(s/n_components)]';
MN_samples=MN_draw*MN_keys;
P_sparse=reshape(MN_samples,n_components,orig_dim);
%spy(P_sparse);

%return random projection matrix
type='sparse';
if (strcmp(type,'gauss'))
    P=P_gauss;
else
    P=P_sparse;
end


