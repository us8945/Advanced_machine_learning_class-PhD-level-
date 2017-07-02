'''
Created on Aug 31, 2016

@author: uri
Use Dual SVMs with Gaussian Kernels to classify SPAM emails
'''
import numpy as np
from numpy import linalg
import numpy
import cvxopt
import math
#from sklearn.cluster.tests.test_k_means import n_samples

class SVM(object):
    def __init__(self, C=1.0, sigma=0.001):
        self.C = C
        self.sigma = sigma
        self.b=0
    
        
    def gaussian_kernel(self,x, y):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (self.sigma ** 2)))

    def kernel_matrix(self,X):
            n_samples, n_features = X.shape
            K = np.zeros((n_samples, n_samples))
            for i, x_i in enumerate(X):
                for j, x_j in enumerate(X):
                    #K[i, j] = linear_kernel(x_i, x_j)
                    K[i, j] = self.gaussian_kernel(x_i, x_j)
            return K
    
    def fit_dual_with_slack(self,X,Y):
        n_samples, n_features = X.shape
    
        K = self.kernel_matrix(X)
    
        P = cvxopt.matrix(np.outer(Y,Y)*K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        
        
        ''' Inequality Constraints: 
            G1 matrix] Lambda is equal or greater than zero
            G2 matrix] Lambda less than or equal to "C" 
        '''
        G1 = np.diag(np.ones(n_samples) * -1)
        G2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((G1, G2)))
        h1 = np.zeros(n_samples)
        h2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((h1, h2)))
        
        ''' Constraint: Sum of (Lambda(i) multiply y(i)) equals to zero
        '''
        A = cvxopt.matrix(Y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        
        cvxopt.solvers.options['show_progress'] = False #disable progress showing by cvxopt solver
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        solution = np.ravel(solution['x'])
        
        '''Support vectors have non zero Lagrange multipliers
        '''
        solution_ind = solution > 1e-5 #True/False array
        sup_v_ind = np.arange(len(solution))[solution_ind]
        
        self.l_multipliers = solution[sup_v_ind] #Lagrange multipliers
        self.sup_v_x = X[sup_v_ind]              #Support vector
        self.sup_v_y = Y[sup_v_ind]              #Support vector label
        
        ''' Calculate intercept 
        '''
        self.b = 0
        for n in range(len(self.l_multipliers)):
            self.b += self.sup_v_y[n]
            self.b -= np.sum(self.l_multipliers * self.sup_v_y * K[sup_v_ind[n],solution_ind])
        self.b = self.b/len(self.l_multipliers)
            
    def predict(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv_x in zip(self.l_multipliers, self.sup_v_y, self.sup_v_x):
                s += a * sv_y * self.gaussian_kernel(X[i], sv_x)
            y_predict[i] = s+self.b
            
        return np.sign(y_predict)
    
    def predict_summary(self,X,Y):
        #n_samples, n_features = X.shape
        correct=0
        missed=0
        result = self.predict(X)
        for i,r in enumerate(result):
            if (r==Y[i]):
                correct=correct+1
            else:
                missed=missed+1
        return correct,missed

if __name__ == '__main__':
    train_df = np.genfromtxt('data/spam_train.data',delimiter=',')
    #train_df = np.genfromtxt('data/spam_train_v0.data',delimiter=',')
    #train_df = train_df[0:10,:] #use for testing
    test_df = np.genfromtxt('data/spam_test.data',delimiter=',')
    validation_df = np.genfromtxt('data/spam_validation.data',delimiter=',')
    
    n_samples, n_features = train_df.shape
    X_train = train_df[:,0:n_features-1]
    Y_train = train_df[:,n_features-1]
    for i,y in enumerate(Y_train): #replace labels zero with -1
        if y==0:
            Y_train[i] = -1    
    
    n_samples, n_features = validation_df.shape
    X_validation = validation_df[:,0:n_features-1]
    Y_validation = validation_df[:,n_features-1]
    for i,y in enumerate(Y_validation): #replace labels zero with -1
        if y==0:
            Y_validation[i] = -1
                
    n_samples, n_features = test_df.shape
    X_test = test_df[:,0:n_features-1]
    Y_test = test_df[:,n_features-1]
    for i,y in enumerate(Y_test): #replace labels zero with -1
        if y==0:
            Y_test[i] = -1
    
    C=[1,10,100,1000,10000]
    Sigma=[.001,.01,.1,1.0,10.0,100.0]

    np.set_printoptions(threshold=np.inf)
    for c in C:
        for s in Sigma:
            svm_mod=SVM(C=c,sigma=s)
            svm_mod.fit_dual_with_slack(X_train,Y_train)
            correct,missed = svm_mod.predict_summary(X_train,Y_train)
            print("Gamma and Sigma",c,s)
            print("Training set:correct and missed",correct,missed)
            
            correct,missed = svm_mod.predict_summary(X_test,Y_test)
            print("Testing set:correct and missed",correct,missed)
            
            correct,missed = svm_mod.predict_summary(X_validation,Y_validation)
            print("Validation set:correct and missed",correct,missed)
    
     
    
                
        
    