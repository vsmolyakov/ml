# ml
Machine Learning Algorithms

### Description

**Markov Chain Monte Carlo**

MCMC methods are useful in sampling from high dimensional distributions by accepting or rejecting samples according to the Metropolis-Hastings (MH) ratio. The figure below shows MCMC-MH sampling results for Gaussian Mixture Model and Bayesian Logistic Regression.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/mcmc/figures/mcmc_merged.png" width = "600" />
</p>

We can see the histogram of accepted samples in the top right plot for the proposal shown in the left. Notice, that in order to maintain high acceptance ration, we would like the proposal distribution to closely match the target distribution. Furthermore, we need to ensure that our samples are drawn from an ergodic markov chain in order to guarantee convergence to the target distribution. While a random walk proposal is used here, to speed up the exploration of the target distribution, we can use other techniques, for example gradient information as used in the Hamiltonian Monte Carlo.

References:  
*C. Andrieu et al, "An Introduction to MCMC for Machine Learning", 2003*  
*R. Neal, "MCMC Using Hamiltonian Dynamics", 2012*

**Gibbs Sampling**

In Gibbs sampling, the samples are drawn from fully conditional distributions. The figure below shows sampling results for a 2D Gaussian distribution and a binary Markov Random Field for image denoising.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gibbs/figures/gibbs_merged.png" width = "600"/>
</p>

The figure on the left shows samples drawn from a 2D Gaussian by alternating x1 ~ p(x1|x2) and x2 ~ p(x2|x1). The more correlated x1 and x2 are the longer it takes to generate samples from the target distribution. The figure on the right shows the Gibbs sampler applied to a noisy binary image modeled as a grid graph, where each pixel is a node and the nodes are connected by weighted edges, where the weight is defined by edge potential psi(xs,xt) = exp(J xs xt), with xs and xt in {-1,+1}. By setting the coupling strength J, we can control the preference for neighboring pixels to have the same state. This is used in image denoising.

A gibbs sampler does not require the design of proposal distribution and the samples are always accepted. However, it can take a long time to converge and alternative methods such as parallel split-merge sampler or variational bayes methods are used to achieve faster approximate results.

References:  
*D. MacKay, "Information Theory, Inference and Learning Algorithms", 2003*

**Gaussian Mixture Models**

Gaussian Mixture Models (GMM) are very popular in approximating many real-world distributions. The figure below shows a graphical model for a GMM and posterior clustering results on synthetic dataset and an image using the Expectation-Maximization (EM) algorithm.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/gmm/figures/gmm_merged.png"/>
</p>

EM algorithm is one of many ways of fitting a GMM. In the E-step, the cluster label assignments are computed, while in the M-step, the means and covariance parameters are updated based on the new labels. EM algorithm results in monotonically increasing objective, however, it can get stuck in local optima and may require several restarts. 

A common modeling question with GMM is how to choose the number of clusters K? Bayesian non-parametric methods extend the GMM to infinite number of clusters using the Dirichlet Process.
Small Variance Asymptotics (SVA) approximations can be used to derive fast algorithms for fitting Dirichlet Process Mixture Models and other bayesian non-parametric extensions.

References:  
*K. Murphy, "Machine Learning: A Probabilistic Perspective", 2012*

**Latent Dirichlet Allocation**

Latent Dirichlet Allocation (LDA) is a topic model that represents each document as a mixture of topics, where a topic is a distribution over words. The objective is to learn the shared topic distributions and their proportions for each document. The figures below show the LDA graphical model, a comparison of EM and variational bayes algorithms on synthetic data and application of online VB to the 20 newsgroups dataset (11,314 docs).


<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/lda/figures/lda_merged.png"/>
</p>

Although both methods are susceptible to local optima, we can see that the EM algorithm achieves higher log-likelihood on synthetically generated data. For real data, we plot the perplexity as a function of iterations and visualize 2 out of K=10 topics.

LDA assumes a bag of words model in which the words are exchangeable and as a result sentence structure is not preserved and only the word counts matter. The LDA topic model associates each word xid with a topic label zid in {1,...,K}. Each document is associated with topic proportions theta_d that could be used to measure document similarity. The topics represented by a Dirichlet distribution are shared across all documents. The hyper-parameters alpha and eta capture our prior knowledge of topic mixtures and multinomials, e.g. from past on-line training of the model.

Topic models have been applied to massive datasets thanks to scalable inference algorithms. In particular, on-line variational methods that sub-sample data into mini-batches such as Stochastic Variational Inference (SVI).

A number of extensions have been proposed for the original LDA model including correlated topic models, a dynamic topic model, supervised LDA and bayesian non-parametric models.

References:  
*D. Blei, A. Ng, M. Jordan, "Latent Dirichlet Allocation", JMLR 2003*  
*M. Hoffman, D. Blei, C. Wang, J. Paisley, "Stochastic Variational Inference", JMLR 2013*

**Stochastic Gradient Descent**

Stochastic Gradient Descent (SGD) methods update parameters based on a sub-set of data. The figure below shows the objective function as a result of SGD updates for Bayesian Logistic Regression.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/sgd/figures/sgd_cost.png" width = "400"/>
</p>

We can see that the gradient fluctuates and on average results in a decreasing objective. SGD updates are important in estimating parameters for massive data-sets. The idea of sub-sampling the data can be applied to scale both Monte Carlo and variational algorithms.

References:  
*R. Bekkerman et. al. "Scaling Up Machine Learning: Parallel and Distributed Approaches", 2011*  

**Hidden Markov Models**

Hidden Markov Models (HMM) can be trained to discover latent states in time-series data. The figure below shows the HMM results trained with EM algorithm on Fitbit data: step count and heart rate.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/hmm/figures/hmm_merged.png"/>
</p>

By fixing the number of states to 2, we can learn the sleep and wake states. The model associates the sleep state with lower mean and variance of the heart-rate and step counts. While it is correct at predicting sleep during the night time, it naively assigns sleep state to periods of inactivity during the day.

The figure on the right shows the price of S&P500 over time. A forward-backward algorithm is used to learn the market state over time based on a Gaussian price observation.

In addition, Bayesian non-parametric methods extend HMMs to include infinite number of states by allocating mass to most probable states according to an HDP prior.

References:  
*C. Bishop, "Pattern Recognition and Machine Learning", 2007*  

**LSTM Language Model**

Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNNs) are capable of sequence learning. A single layer LSTM network was trained on sentences from the works of Shakespeare in a corpus of over a million characters. Each sentence is represented as a sequence of characters and the network learns a language model by predicting the probability of the next character given the input sequence.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/lstm/figures/language_model_merged.png"/>
</p>

The figure above shows the schematic of the LSTM network (left) and Shakespeare (right) generated by LSTM after only 10 training epochs. Increasing the size of the RNN and training for longer period of time can further improve the language model.


**Misc**

There are a number of fun demos in the misc category: estimating pi using monte carlo, randomized monte carlo for self-avoiding random walks, Gaussian kernel density estimator, importance and rejection sampling, random forrest and SGD neuron classifier.

<p align="center">
<img src="https://github.com/vsmolyakov/ml/blob/master/misc/figures/misc_merged.png"/>
</p>
 
### Dependencies

Matlab 2014a  
Python 2.7