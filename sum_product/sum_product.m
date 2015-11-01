function [c, m_f, m_b] = sum_product(A, sigma_b, sigma_g, mu_g, px0, y)
% A: transition matrix = 
%     [ p(xt+1 = b | xt = b)    p(xt+1 = b | xt = g) ]
%     [ p(xt+1 = g | xt = b)    p(xt+1 = g | xt = g) ]
% 
% sigma_b: std dev of yt conditioned on xt = b
% mu_g, sigma_g: mean, std dev of yt conditioned on xt = g
% px0: initial state probability
% y: emissions
% c: marginals, i.e. normalized gamma
% m_f, m_b: forward and backward messages

N = length(y);

m_f = zeros(size(px0,1), N);
m_b = m_f;

% messages from observed to hidden nodes: Pr{y_t|x_t}
B = [1/(sqrt(2*pi)*sigma_b)*exp(-y.^2/(2*sigma_b^2)) ...
     1/(sqrt(2*pi)*sigma_g)*exp(-(y-mu_g).^2/(2*sigma_g^2))]';

% forward pass
m_f(:,1) = px0;
for i=1:N-1  %note: magnitude drops rapidly due to product
    m_f(:,i+1) = sum(A * diag(m_f(:,i) .* B(:,i)), 2);
end

% backward pass
m_b(:,N) = 1;
for i=N-1:-1:1 %note: magnitude drops rapidly due to product
    m_b(:,i) = sum(diag(B(:,i+1) .* m_b(:,i+1)) * A, 1)';
end

% compute marginals
c = m_f .* m_b .* B;
c = c / sum(c(:,1));
