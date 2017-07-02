'''
Created on Aug 31, 2016

@author: uri
Use data set (mystery.data) that contains four numeric attributes per row and the fifth entry is the class variable (either + or -). 
Find a perfect classifer for this data set using Primal approach for SVM.
Use cvxopt linbrary for quadratic optimization solution.
'''
import numpy as np
import numpy
import cvxopt
import math
#from sklearn.cluster.tests.test_k_means import n_samples


def compute_primal(X,Y):
    n_samples, n_features = X.shape
    P = cvxopt.matrix(np.identity(n_features)) ##w*P*w P-identity matrix
    q = cvxopt.matrix(np.zeros(n_features))
    Y1=(-1)*Y
    G1=X
    for i,y in enumerate(Y1):
        G1[i,:]=G1[i,:]*y
    G = cvxopt.matrix(G1)
    h1=np.ones(n_samples)
    h1=h1*(-1)
    h = cvxopt.matrix(h1)

    solution = cvxopt.solvers.qp(P, q, G, h)#, A, b)
    solution = np.ravel(solution['x'])    
    return solution

if __name__ == '__main__':
    train_df = np.genfromtxt('data/mystery.data',delimiter=',')
    X = train_df[:,0:4]
    Y = train_df[:,4]
    n_samples, n_features = X.shape
    X1=np.ones((n_samples,n_features+1))
    X1[:,:-1]=X #Add 1 to the end of every raw to mimic B coefficient
    X2=X1*X1
    multipliers=compute_primal(X2,Y)
    print(multipliers)
     
    
                
        
    