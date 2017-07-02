'''
Created on Aug 31, 2016

@author: uri
Use Primal SVM with slack to classify SPAM email
'''
import numpy as np
import numpy
import cvxopt
import math
from copy import deepcopy



def compute_primal(X,Y):
    n_samples, n_features = X.shape
    P = cvxopt.matrix(np.identity(n_features)) ##w*P*w P-identity matrix
    q = cvxopt.matrix(np.zeros(n_features))
    Y1=(-1)*Y
    G1=deepcopy(X)
    for i,y in enumerate(Y1):
        G1[i,:]=G1[i,:]*y
    G = cvxopt.matrix(G1)
    h1=np.ones(n_samples)
    h1=h1*(-1)
    h = cvxopt.matrix(h1)

    solution = cvxopt.solvers.qp(P, q, G, h)#, A, b)
    solution = np.ravel(solution['x'])    
    return solution

def compute_primal_with_slack(X,Y,C):
    '''
    min (1/2 W^2 + C*gammas)
    S.T.  -1(Y*X*W + gammas) <= -1
           -1*gammas <= 0
    '''
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
    #print(G4.shape)
    #print(G5.shape)
    G6=np.concatenate([G4,G5],axis=1)
    print(G6.shape)
    
    G=np.concatenate([G3,G6],axis=0)
    print(G.shape)
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

def predict_summary(X,Y,multipliers):
    n_samples, n_features = X.shape
    correct=0
    missed=0
    if n_features != multipliers.size:
        print("Mismatch in number of features and multipliers",n_features,multipliers.size)
    result = np.dot(X,multipliers)
    #print("Results",result)
    for i,r in enumerate(result):
        if (r>0 and Y[i]==1) or (r<0 and Y[i]==-1):
            correct=correct+1
        else:
            missed=missed+1
    return correct,missed

if __name__ == '__main__':
    train_df = np.genfromtxt('data/spam_train.data',delimiter=',')
    #train_df = train_df[0:300,:] #use for testing
    test_df = np.genfromtxt('data/spam_test.data',delimiter=',')
    validation_df = np.genfromtxt('data/spam_validation.data',delimiter=',')
    n_samples, n_features = train_df.shape
    X_train = train_df[:,0:n_features-1]
    Y_train = train_df[:,n_features-1]
    
    for i,y in enumerate(Y_train): #replace labels zero with -1
        if y==0:
            Y_train[i] = -1
    
    
    C=[1.0,10.0,100.0,1000.0,10000.0]
    n_samples, n_features = X_train.shape
    X1_train=np.ones((n_samples,n_features+1))
    X1_train[:,:-1]=X_train #Add 1 to the end of every raw to mimic B coefficient
    
    n_samples, n_features = validation_df.shape
    X_validation = validation_df[:,0:n_features-1]
    Y_validation = validation_df[:,n_features-1]
    for i,y in enumerate(Y_validation): #replace labels zero with -1
        if y==0:
            Y_validation[i] = -1
            
    n_samples, n_features = X_validation.shape        
    X1_validation=np.ones((n_samples,n_features+1))
    X1_validation[:,:-1]=X_validation #Add 1 to the end of every raw to mimic B coefficient
    
    n_samples, n_features = test_df.shape
    X_test = test_df[:,0:n_features-1]
    Y_test = test_df[:,n_features-1]
    for i,y in enumerate(Y_test): #replace labels zero with -1
        if y==0:
            Y_test[i] = -1
    n_samples, n_features = X_test.shape
    X1_test=np.ones((n_samples,n_features+1))
    X1_test[:,:-1]=X_test #Add 1 to the end of every raw to mimic B coefficient
    
    for c in C:
        multipliers=compute_primal_with_slack(X1_train,Y_train,c)
        multipliers = multipliers[0:n_features+1] #filter out slack multipliers
        np.set_printoptions(threshold=np.inf)
        correct,missed = predict_summary(X1_train,Y_train,multipliers)
        print("Training set gamma:",c,"accuracy:",(correct/(correct+missed)))
        correct,missed = predict_summary(X1_validation,Y_validation,multipliers)
        print("Validation set gamma:",c,"accuracy:",(correct/(correct+missed)))
        correct,missed = predict_summary(X1_test,Y_test,multipliers)
        print("Testing set gamma:",c,"accuracy:",(correct/(correct+missed)))
     
    
                
        
    