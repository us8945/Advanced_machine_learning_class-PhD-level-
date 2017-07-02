'''
Created on Aug 31, 2016

@author: uri
Implement nearest-k method to classify SPAM email
'''
import numpy as np
from numpy import linalg
import numpy
import cvxopt
import math
from matplotlib.pyplot import axis
#from sklearn.cluster.tests.test_k_means import n_samples

class K_nearest(object):
    def __init__(self,X,Y,k=1):
        self.k=k
        self.X=X
        self.Y=Y
        self._normalize()
    
    def _fit_distance(self,x_norm):
        distances=[]
        for x_train in self.X_norm:
            dist=numpy.linalg.norm(x_train-x_norm)
            distances.append(dist)
        return np.array(distances)
    
    def _normalize(self):
        self.mean=np.mean(self.X,axis=0) #array of column means
        self.std=1/np.std(self.X,axis=0)
        self.X_norm=(self.X-self.mean)*self.std
    
    def predict(self,x):
        x_norm=(x-self.mean)*self.std
        distances = self._fit_distance(x_norm)
        distances=distances.reshape(len(distances),1)
        y=self.Y.reshape(len(self.Y),1)
        #print(distances.shape,type(distances))
        #print(y.shape,type(y))
        dist_result=np.hstack((distances,y))
        dist_result = dist_result[dist_result[:,0].argsort(axis=0)] #sort by first column/sort by distance
        predition=0
        for i,d in enumerate(dist_result):
            if i==self.k:
                return predition
            predition=predition+d[1] #aggregate labels from nearest neighbors 
    
    
    def predict_summary(self,X,Y):
        #n_samples, n_features = X.shape
        correct=0
        missed=0
        
        for i,x in enumerate(X):
            r = self.predict(x)
            if r==0:
                r=np.random.rand(1)[0]-0.5 #break tie randomly
            if (np.sign(r) == np.sign(Y[i])):
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
    
    k_nrst = K_nearest(X=X_train,Y=Y_train,k=1)
    correct,missed = k_nrst.predict_summary(X_test, Y_test)
    print("Test set prediction, k-nearerst=1",correct,missed)
    correct,missed = k_nrst.predict_summary(X_validation, Y_validation)
    print("Validation set prediction, k-nearerst=1",correct,missed)

    k_nrst = K_nearest(X=X_train,Y=Y_train,k=2)
    correct,missed = k_nrst.predict_summary(X_test, Y_test)
    print("Test set prediction, k-nearerst=2",correct,missed)
    correct,missed = k_nrst.predict_summary(X_validation, Y_validation)
    print("Validation set prediction, k-nearerst=2",correct,missed)
    
    k_nrst = K_nearest(X=X_train,Y=Y_train,k=3)
    correct,missed = k_nrst.predict_summary(X_test, Y_test)
    print("Test set prediction, k-nearerst=3",correct,missed)
    correct,missed = k_nrst.predict_summary(X_validation, Y_validation)
    print("Validation set prediction, k-nearerst=3",correct,missed)
    
                
        
    