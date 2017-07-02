'''
Created on Aug 31, 2016

@author: uri
Implement PCA and SVM for feature selection odf SPAM classification problem
'''
import numpy as np
import numpy
import cvxopt
import math
from copy import deepcopy
from numpy import std

def compute_primal_with_slack(X_i,Y,C):
    '''
    min (1/2 W^2 + C*gammas)
    S.T.  -1(Y*X*W + gammas) <= -1
           -1*gammas <= 0
    '''
    X=deepcopy(X_i)
    n_samples, n_features = X.shape
    X1=np.ones((n_samples,n_features+1))
    X1[:,:-1]=X #Add 1 to the end of every raw to mimic B coefficient
    X=X1
    
    n_samples, n_features = X.shape
    np_P=np.identity(n_features+n_samples)
    for i in range(n_features,n_features+n_samples):
        np_P[i,i]=0 ##assign zero for gamma
    P = cvxopt.matrix(np_P) ##w*P*w P-identity matrix
    
    np_q=np.ones(n_features+n_samples)
    np_q = np_q*C                    #multiply by slack
    for i in range(n_features):
        np_q[i]=0                  ##assign zero for W
    q = cvxopt.matrix(np_q)
    
    Y1=(-1)*Y
    G1=deepcopy(X)
    for i,y in enumerate(Y1):
        G1[i,:]=G1[i,:]*y
    
    '''Add constraints for gamma:
       1) Add -1 column as number of samples to each record
       2) Add record to the end with zeros as n_feature, -1 as num_samples [0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,...]
    '''
    G2=np.ones(n_samples*n_samples).reshape(n_samples,n_samples)
    G2=G2*(-1)
    
    G3=np.concatenate([G1,G2],axis=1) # combine G1 and G2 by columns
    #print(G3.shape)
    
    ''' Add gamma >= 0 constraints
    '''
    G4=np.zeros(n_features*n_samples).reshape(n_samples,n_features)
    G5=np.identity(n_samples)
    G5=G5*(-1)
    G6=np.concatenate([G4,G5],axis=1)
    
    G=np.concatenate([G3,G6],axis=0)
    G = cvxopt.matrix(G) # combine G1 and G2 by columns
    
    ''' First n_sample constraints are <=-1
        Second n_sample constraints are <=0
    '''                  
    h1=np.ones(n_samples)
    h1=h1*(-1)
    h2=np.zeros(n_samples)
    h=np.concatenate([h1,h2])
    h = cvxopt.matrix(h)

    solution = cvxopt.solvers.qp(P, q, G, h)#, A, b)
    solution = np.ravel(solution['x'])    
    return solution

def predict_summary(X_i,Y,multipliers):
    n_samples, n_features = X_i.shape
    X=deepcopy(X_i)
    n_samples, n_features = X.shape
    X1=np.ones((n_samples,n_features+1))
    X1[:,:-1]=X #Add 1 to the end of every raw to mimic B coefficient
    X=X1
    if (n_features+1) != multipliers.size:
        print("Mismatch in number of features and multipliers",n_features,multipliers.size)
    
    result = np.dot(X,multipliers)*Y
    '''If correctly predicted result of multiplying by Y is positive
       Count number of results > 0
    '''
    correct = len(np.where(result>0)[0])
    missed=n_samples-correct
    return correct,missed

def normalize(X,i_mean):
    ''' Perform matrix wise normalization
    '''
    if i_mean is None:
        mean = np.mean(X,axis=None)
    else:
        mean=i_mean
    
    X_norm=(X-mean)
    return X_norm, mean

def calc_pca(X,k):
    XX=np.dot(X.T,X)
    eig_val,eig_vec = np.linalg.eig(XX)
    '''eigvector is column wise, not record wise !!!
       Sort by eig-value to guarantee order. Numpy doesn't guarantee order
    '''
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    matrix_w=eig_pairs[0][1].reshape(len(eig_pairs),1)
    for i in range(1,k):
        matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(len(eig_pairs),1)))
    return matrix_w

def calc_pca_top_eig_val(X,k):
    '''Return top k higest eigenvalues
    '''
    XX=np.dot(X.T,X)
    eig_val,eig_vec = np.linalg.eig(XX)
    return np.sort(eig_val)[::-1][0:k,]


def load_and_normalize_data():
    train_df = np.genfromtxt('data/spam_train.data',delimiter=',')
    #train_df = train_df[0:500,:] #use for testing
    test_df = np.genfromtxt('data/spam_test.data',delimiter=',')
    validation_df = np.genfromtxt('data/spam_validation.data',delimiter=',')
    n_samples, n_features = train_df.shape
    X_train = train_df[:,0:n_features-1]
    Y_train = train_df[:,n_features-1]
    Y_train[np.where(Y_train==0)]=-1 #replace zero with -1
    X_train,x_mean = normalize(X_train, None)
    
    n_samples, n_features = validation_df.shape
    X_validation = validation_df[:,0:n_features-1]
    Y_validation = validation_df[:,n_features-1]
    Y_validation[np.where(Y_validation==0)]=-1 #replace zero with -1
    X_validation,x_mean=normalize(X_validation, x_mean)
    
    n_samples, n_features = test_df.shape
    X_test = test_df[:,0:n_features-1]
    X_test,x_mean=normalize(X_test, x_mean)
    Y_test = test_df[:,n_features-1]
    Y_test[np.where(Y_test==0)]=-1 #replace zero with -1
    
    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

if __name__ == '__main__':
    X_train,Y_train,X_validation,Y_validation,X_test,Y_test=load_and_normalize_data()
    '''
    Problem 1-1: calculate top six eigenvalues 
    '''
    top_eigenvalues = calc_pca_top_eig_val(X_train,6)
    print("Top six eigenvalues")
    print(top_eigenvalues)
    
    all_eigenvalues = calc_pca_top_eig_val(X_train,57)
    print("All eigenvalues")
    print(all_eigenvalues)
    
    
    C=[1.0,10.0,100.0,1000.0]
    pca_vec=[1,2,3,4,5,6]
    for c in C:
        #multipliers=compute_primal_with_slack(X_train,Y_train,c)
        for num_vec in pca_vec:
            W=calc_pca(X_train, num_vec)
            x_train_transform=X_train.dot(W)
            x_test_transform=X_test.dot(W)
            x_validation_transform=X_validation.dot(W)
            multipliers=compute_primal_with_slack(x_train_transform,Y_train,c)
            multipliers = multipliers[0:num_vec+1] #filter out slack multipliers
            np.set_printoptions(threshold=np.inf)
            correct,missed = predict_summary(x_train_transform,Y_train,multipliers)
            print("Training set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
            correct,missed = predict_summary(x_validation_transform,Y_validation,multipliers)
            print("Validation set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
            correct,missed = predict_summary(x_test_transform,Y_test,multipliers)
            print("Testing set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
     
    
                
        
    