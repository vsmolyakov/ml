import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris

np.random.seed(0)

class sgdlr:
    
    def __init__(self):
        
        self.num_iter = 100
        self.lmbda = 1e-9
        
        self.tau0 = 10
        self.kappa = 1
        
        self.batchsize = 200
        
    def fit(self,X,y):

        theta = np.random.randn(X.shape[1],1)
        #learning rate
        eta = np.zeros((self.num_iter,1))
        for i in range(self.num_iter):
            eta[i] = (self.tau0+i)**(-self.kappa)        
                
        #divide data in batches        
        batchdata, batchlabels = self.make_batches(X,y,self.batchsize)                    
        num_batches = batchdata.shape[0]
        num_updates = 0
        
        J_rec = np.zeros((self.num_iter * num_batches,1))
        t_rec = np.zeros((self.num_iter * num_batches,1))
        
        for itr in range(self.num_iter):
            for b in range(num_batches):
                Xb = batchdata[b]
                yb = batchlabels[b]
                J_cost, J_grad = self.lr_objective(theta, Xb, yb, self.lmbda)
                theta = theta - eta[itr]*(num_batches*J_grad)                
                J_rec[num_updates] = J_cost
                t_rec[num_updates] = np.linalg.norm(theta,2)
                num_updates = num_updates+1
            print "iteration %d, cost: %f" %(itr, J_cost)
        
        y_pred = 2*(self.sigmoid(X.dot(theta)) > 0.5) - 1
        y_err = np.size(np.where(y_pred - y)[0])/float(y.shape[0])
        print "classification error: %f" %(y_err)        

        self.generate_plots(X, J_rec, t_rec, theta)
        return theta        
    
    def make_batches(self,X,y,batchsize):
        n = X.shape[0]
        d = X.shape[1]
        num_batches = int(np.ceil(n/batchsize))
        groups = np.tile(range(num_batches),batchsize)
        batchdata=np.zeros((num_batches,batchsize,d))
        batchlabels=np.zeros((num_batches,batchsize,1))
        for i in range(num_batches):
            batchdata[i,:,:] = X[groups==i,:]
            batchlabels[i,:] = y[groups==i]
            
        return batchdata, batchlabels
            
    def lr_objective(self,theta,X,y,lmbda):
        
        n = y.shape[0]
        y01 = (y+1)/2.0
        
        #compute the objective
        mu = self.sigmoid(X.dot(theta))
    
        #bound away from 0 and 1
        eps = np.finfo(float).eps   
        mu = np.maximum(mu,eps)
        mu = np.minimum(mu,1-eps)    
        
        #compute cost
        cost = -(1/n)*np.sum(y01*np.log(mu)+(1-y01)*np.log(1-mu))+np.sum(lmbda*theta*theta)
        
        #compute the gradient of the lr objective
        grad = X.T.dot(mu-y01) + 2*lmbda*theta
        
        #compute the Hessian of the lr objective
        #H = X.T.dot(np.diag(np.diag( mu*(1-mu) ))).dot(X) + 2*lmbda*np.eye(np.size(theta))
        
        return cost, grad
        
                    
    def sigmoid(self,a):
        return 1/(1+np.exp(-a))
    
    def generate_plots(self,X,J_rec,t_rec,theta):
                        
        plt.figure()
        plt.plot(J_rec)
        plt.title("logistic regression")
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
        
        plt.figure()
        plt.plot(t_rec)
        plt.title("logistic regression")
        plt.xlabel('iterations')
        plt.ylabel('theta l2 norm')
        plt.show()                
        
        plt.figure()
        x1 = np.linspace(np.min(X[:,0])-1,np.max(X[:,0])+1,10)
        plt.scatter(X[:,0], X[:,1])
        plt.plot(x1, -(theta[0]/theta[1])*x1)
        plt.title('logistic regression')
        plt.grid(True)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        
    
    def generate_data(self):
        
        n = 1000
        mu1 = np.array([1,1])
        mu2 = np.array([-1,-1])        
        pik = np.array([0.4,0.6])
        
        X = np.zeros((n,2))
        y = np.zeros((n,1))
        
        for i in range(1,n):
            u = np.random.rand()
            idx = np.where(u < np.cumsum(pik))[0]                            
            
            if (len(idx)==1):
                X[i,:] = np.random.randn(1,2) + mu1
                y[i] = 1
            else:
                X[i,:] = np.random.randn(1,2) + mu2
                y[i] = -1
                
        return X, y    
        
                

if __name__ == "__main__":
    
    #iris = load_iris()
    #X = iris.data
    #y = iris.target
    
    sgd = sgdlr()
    X, y = sgd.generate_data()
    theta = sgd.fit(X,y)
    
    