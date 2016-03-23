# ml
Machine Learning Algorithms

### Description

Markov Chain Monte Carlo

MCMC methods are used in sampling from high dimensional distributions by accepting or rejecting samples according to a Metropolis-Hastings (MH) ratio. The figure below illustrates MCMC-MH for a bimodal 2D Gaussian distribution.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/mcmc/figures/mcmc_gaussmix2d.png?raw=true"/>
</p>

To maintain high acceptance ratio, the proposal distribution should closely match the target distribution. It's important to maintain ergodicity to guarantee that the accepted samples drawn from the proposal distribution converge to the target distribution. To speed up the exploration of the target distribution, the random walk proposal can be replaced with methods that use gradient information such as Hamiltonian Monte Carlo.

Reference: Andrieu C. et al, "An Introduction to MCMC for Machine Learning", 2003

Gibbs Sampling

In Gibbs sampling, the samples are drawn from a fully conditional distribution. The figure applies Gibbs sampling to image denoising using a binary Markov Random Field (MRF) model.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gibbs/figures/mean_gibbs.png?raw=true"/>
</p>

Gaussian Mixture Models

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gmm/figures/gmm_clusters.png?raw=true"/>
</p>

Latent Dirichlet Allocation

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/lda/figures/lda.png?raw=true"/>
</p>

Stochastic Gradient Descent

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/sgd/figures/sgd_cost.png?raw=true"/>
</p>

Hidden Markov Models

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/hmm/figures/sp500.png?raw=true"/>
</p>

Misc

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/misc/figures/density_est.png" width="50%"/>
</p>


### References

 
### Dependencies

Matlab 2014a
