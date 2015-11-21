import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier
    Parameters
    ----------
    eta: learning rate
    n_iter: number of passes over training data

    Attributes
    ----------
    w_: weights after fitting
    errors_: misclassification errors
    shuffle: shuffles training data every epoch
    random_state: random state for shuffling
    """
     
    def __init__(self, eta=0.01, n_iter = 10, shuffle=True, random_state=None):
        self.eta=eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)
            
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
            
        return self
    
    def partial_fit(self, X, y):
        """ Fit training data without reinitializing the weights """
        if not self.w_initiliazed:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        
        return self
        
    def _shuffle(self, X, y):
        """ Shuffle training data """
        r = np.random.permutation(len(y))
        return X[r], y[r]
        
    
    def _initialize_weights(self,m):
        """ Initialize weights to zeros """
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """ Apply Adaline learning rule to update the weights """
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self,X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    
    def activation(self,X):
        """ Compute linear activation """
        return self.net_input(X)
    
    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
 
def plot_decision_regions(X, y, classifier, resolution=0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,Z,alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y = X[y==cl,1], alpha=0.8, c=cmap(idx), marker = markers[idx], label=cl)
       

if __name__ == "__main__":

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    
    y = df.iloc[0:100,4].values
    y = np.where(y == 'Iris-setosa', -1,1)
    X = df.iloc[0:100, [0,2]].values
 
    #standardization
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()     

    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    
    plot_decision_regions(X_std,y,classifier=ada)
    plt.title('Adaline - SGD')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()
    
            
                            