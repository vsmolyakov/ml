%% Estimate Pi using Monte Carlo
%  A=pi r^2, pi_hat = A / r^2
%  x,y ~ Unif(-r,r)
%  I=int_a^b h(x) dx = int_a^b w(x)p(x) dx, where w(x)=h(x)(b-a) and
%  p(x)=1/(b-a)

% Reference: 
% http://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/mcmc.pdf

%% Monte Carlo Algorithm

r=1;
num_iter=5e3;
x=unifrnd(-r,r,num_iter,1);
y=unifrnd(-r,r,num_iter,1);
rs2=x.^2+y.^2;
inside=(rs2<=r^2);
samples=(2*r)*(2*r)*inside; %(bx-ax)*(by-ay)*I[x^2+y^2<=r^2]
I_hat=mean(samples);
pi_hat=I_hat/r^2
pi_hat_se=sqrt(var(samples))/sqrt(num_iter)

%% generate plots
figure(1);
outside=~inside;
plot(x(inside),y(inside),'bo'); hold on;
plot(x(outside),y(outside),'rx'); axis square;

