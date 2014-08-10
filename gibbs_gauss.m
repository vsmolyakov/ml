%% Gibbs sampler for multivariate Gaussian

function samples = gibbs_gauss(mu, Sigma, xinit, Nsamples)
D=length(mu);
samples=zeros(Nsamples,D);
x=xinit(:)';
for s=1:Nsamples
    for i=1:D
        [muAgivenB, sigmaAgivenB] = gauss_conditional(mu,Sigma,i,x); %leave-one-out
        x(i)=normrnd(muAgivenB,sqrt(sigmaAgivenB)); %sample from conditional
    end
    samples(s,:)=x;
end
end


function [muAgivenB, sigmaAgivenB] = gauss_conditional(mu,Sigma,a,x)
% mu1|2 = mu1 + S12 inv(S22) (x2-mu2)
% S1|2  = S11 - S12 inv(S22) S21

D=length(mu);
b=setdiff(1:D,a);
muA=mu(a); muB=mu(b);
SAA=Sigma(a,a);
SAB=Sigma(a,b);
SBB=Sigma(b,b);
SBBinv=inv(SBB);
muAgivenB=mu(a)+SAB*SBBinv*(x(b)-mu(b));
sigmaAgivenB=SAA-SAB*SBBinv*SAB';

end