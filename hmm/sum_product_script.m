%% Sum-Product Algorithm applied to HMM with time-series market data
%  HMM model: state x_t \in {'g','b'}:
%  x_t = Ax_t-1 + w, A = [q 1-q; 1-q q];
%  observation y_t:
%  y_t|x_t~N(0,sigma_b^2) when x_t = 'b', N(mu_g,sigma_g^2) when x_t = 'g'

close all; clear all

%% dataset
load sp500.dat

figure; 
plot(sp500); title('time series price data of S&P500'); grid on;
xlabel('week number'); ylabel('price');

%% HMM parameters

q = 0;                   %self-transition probability
A = [q, 1-q; 1-q, q];    %transition matrix
px0 = [0.5; 0.5];        %init probability
sigma_b = 30;            %y_t|x_t=b std dev
sigma_g = 20; mu_g = 20; %y_t|x_t=g std dev and mean
y = diff(sp500);         %observation = difference in cons prices

%% sum-product algorithm
[c, m_f, m_b] = sum_product(A, sigma_b, sigma_g, mu_g, px0, y);

%% generate plots
figure;
plot(c(1,:),'LineWidth',1.5); title('MAP estimate of SP500 state');
xlabel('week'); ylabel('Pr{xt=g}'); legend('xt=g');

figure;
semilogy(m_f(1,:),'LineWidth',1.5); grid on; 
title('Magnitude decrease of forward messages in the sum-product algorithm');
xlabel('week'); ylabel('log(m_f)'); legend('xt=g');
