# ml
Machine Learning Algorithms

### Description

**Markov Chain Monte Carlo**

MCMC methods are useful in sampling from high dimensional distributions by accepting or rejecting samples according to the Metropolis-Hastings (MH) ratio. The figure below shows MCMC-MH sampling results for Gaussian mixture and Bayesian logistic regression.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/mcmc/figures/mcmc_merged.png" width = "600" />
</p>

We can see the histogram of accepted samples in the top right plot for the proposal shown in the left. Notice, that in order to maintain high acceptance ration, we would like the proposal distribution to closely match the target distribution. Furthermore, we need to ensure that our samples are drawn from an ergodic markov chain in order to guarantee convergence to the target distribution. While a random walk proposal is used here, to speed up the exploration of the target distribution, we can use other techniques, for example gradient information as used in the Hamiltonian Monte Carlo.

*Reference: Andrieu C. et al, "An Introduction to MCMC for Machine Learning", 2003*

**Gibbs Sampling**

In Gibbs sampling, the samples are drawn from fully conditional distributions. The figure below shows sampling results for a 2D Gaussian distribution and a binary Markov Random Field for image denoising.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gibbs/figures/gibbs_merged.png" width = "600"/>
</p>

The figure on the left shows samples drawn from a 2D Gaussian by alternating x1 ~ p(x1|x2) and x2 ~ p(x2|x1). The more correlated x1 and x2 are the longer it takes to generate samples from the target distribution. The figure on the right shows the Gibbs sampler applied to a noisy binary image modeled as a graph, where each pixel is a node and the nodes are connected by weighted edges, where the weight is defined by an edge potential psi(xs,xt) = exp(J*xs*xt) , xs and xt \in {-1,+1}. By setting the coupling strength J, we can control the preference for neighboring pixels to have the same state. This helps in image denoising.

A gibbs sampler does not require the design of proposal distribution and the samples are always accepted. However, it can take a long time to converge and alternative methods such as parallel split-merge sampler or variational bayes methods are used to achieve faster approximate results.

*Reference: D. MacKay, "Information Theory, Inference and Learning Algorithms", 2003*

**Gaussian Mixture Models**

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gmm/figures/gmm_clusters.png"/>
</p>

**Latent Dirichlet Allocation**

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/lda/figures/lda.png"/>
</p>

**Stochastic Gradient Descent**

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/sgd/figures/sgd_cost.png"/>
</p>

**Hidden Markov Models**

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/hmm/figures/sp500.png"/>
</p>

**Misc**

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/misc/figures/density_est.png"/>
</p>


### References

 
### Dependencies

Matlab 2014a
