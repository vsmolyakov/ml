%% Density Estimator
% Histogram and Gaussian Kernel estimators used in analysis of RNA-seq data
% for flux estimation of a T7 promoter

%% parameters
G=1e9; %length of genome, base pairs (bp)
C=1e3; %number of unique molecules in the library, molecules
L=100; %length of a read, bp
N=1e6; %number of reads, L bp long
M=1e4; %number of unique read sequences, bp
LN=1e3; %total length of assembled / mapped RNA-seq reads, i.e. output of string or de Bruijn graph, bp
FDR=0.05; %false discovery rate (alpha)

%% uniform sampling (poisson model)
lambda=(N*L)/(G); %expected number of bases covered
C_est=M/(1-exp(-lambda)); %library size estimate
C_cvrg=G-G*exp(-lambda); %base covarage
N_gaps=N*exp(-lambda); %number of gaps (uncovered bases)

%% gamma prior sampling (negative binomial model)
%X = the number of failures before rth success
k=0.5; %dispersion parameter (fit to data)
p=lambda/(lambda+1/k); %success probability
r=1/k; %number of successes
nbin_pdf=nbinpdf(1:L,r,p);
%plot(nbin_pdf)
%C_est=M/(1-NB(0;lambda,k));

%% RNAP binding data (RNA-seq)
data=nbinrnd(r,p,1,LN); %generate negative binomial binding data
bin_delta=1; %bin width (assume constant for all bins)
binrange=1:bin_delta:max(data); %end points for each bin
[bincounts, ind]=histc(data,binrange); %bincounts: counts of each bin, ind: sorted by max counts
[nelements,bincenters]=hist(data,binrange); %histogram plot

%% density estimation 
%P=integral_R p(x)dx where x is in R^3
%p(x)=K/(NxV), where K=number of points that fall into region R,
%N=total number of points, V=volume of the region R

%histogram density estimator (with smoothing parameter bin_delta):
%histogram projects the data to R^1
rnap_density_est1=bincounts/(sum(bincounts)*bin_delta);
figure; plot(rnap_density_est1,'-b','LineWidth',1.5);
title('RNA-seq density estimate based on negative binomial sampling model');
xlabel('read length, [base pairs]'); ylabel('density'); grid on; hold on;

%Gaussian kernel density estimator (with smoothing parameter h):
%sum N gaussians centered at each datapoint, parameterized by common standard deviation h
x_dim=1; %dimension of x
h=10; %standard deviation
rnap_density_support=linspace(0,max(data),L);
rnap_density_est2=0; %zeros(1,length(rnap_density_support));

for i=1:sum(bincounts)
    rnap_density_est2=rnap_density_est2+(1/(2*pi*h^2)^(x_dim/2))*exp(-(rnap_density_support-data(i)).^2/(2*h^2));
end
rnap_density_est2=rnap_density_est2/sum(rnap_density_est2);
plot(rnap_density_support, rnap_density_est2,'-r','LineWidth',1.5);
legend('histogram','gaussian kernel'); xlim([0,max(rnap_density_support)]);
