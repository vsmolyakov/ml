# ml
Machine Learning Algorithms

### Description

**Markov Chain Monte Carlo**

MCMC methods are useful in sampling from high dimensional distributions by accepting or rejecting samples according to the Metropolis-Hastings (MH) ratio. The figure below shows MCMC-MH sampling results for Gaussian mixture and bayesian logistic regression.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/mcmc/figures/mcmc_merged.png" width = "600" />
</p>

We can see the histogram of accepted samples in the top right plot for the proposal shown in the left.
Notice, that in order to maintain high acceptance ration, we would like the proposal distribution to closely match the target distribution. Furthermore, we need to ensure that our samples are drawn from an ergodic markov chain in order to guarantee convergence to the target distribution. While a random walk proposal is used here, to speed up the exploration of the target distribution, we can use other techniques, for example gradient information as used in the Hamiltonian Monte Carlo.

Reference: Andrieu C. et al, "An Introduction to MCMC for Machine Learning", 2003

Gibbs Sampling

In Gibbs sampling, the samples are drawn from a fully conditional distribution. The figure applies Gibbs sampling to image denoising using a binary Markov Random Field (MRF) model.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gibbs/figures/mean_gibbs.png"/>
</p>

Gaussian Mixture Models

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gmm/figures/gmm_clusters.png"/>
</p>

Latent Dirichlet Allocation

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/lda/figures/lda.png"/>
</p>

Stochastic Gradient Descent

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/sgd/figures/sgd_cost.png"/>
</p>

Hidden Markov Models

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/hmm/figures/sp500.png"/>
</p>

Misc

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/misc/figures/density_est.png"/>
</p>


### References

 
### Dependencies

Matlab 2014a
