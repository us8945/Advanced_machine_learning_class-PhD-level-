'''
Created on Aug 31, 2016

@author: uri
Use data set (mystery.data) that contains four numeric attributes per row and the fifth entry is the class variable (either + or -). 
Find a perfect classifer for this data set using dual approach for SVM.
'''
import numpy as np
import numpy
import cvxopt
import math
#from sklearn.cluster.tests.test_k_means import n_samples

def construct_predictor(X, Y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > 1e-5

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = Y[support_vector_indices]
        weights = np.array([0,0,0,0])
        for i,x in enumerate(support_vectors):
            weights = weights+support_multipliers[i]*support_vector_labels[i]*x
        
        #print(weights)
        b=Y[0]*math.sqrt(np.dot(weights,weights)) - np.dot(X[0],weights)
        print(b)
        b=Y[1]*math.sqrt(np.dot(weights,weights)) - np.dot(X[1],weights)
        print(b)
        #print(support_vector_indices)
        #print(lagrange_multipliers)
        return weights,b

def compute_multipliers(X, Y , c):
    #http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        n_samples, n_features = X.shape

        K = gram_matrix(X)
        print(K.shape)
        print(np.outer(Y,Y).shape)

        P = cvxopt.matrix(np.outer(Y,Y)*K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        
        
        ''' Constraint: Lambda is equal or greater than zero
        '''
        G = cvxopt.matrix(np.diag(np.ones(n_samples)*(-1)))
        h = cvxopt.matrix(np.zeros(n_samples))
        
        ''' Constraint: Sum of (Lambda(i) multiply y(i)) equals to zero
        '''
        A = cvxopt.matrix(Y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        solution = np.ravel(solution['x'])
        
        '''Support vectors have non zero Lagrange multipliers
        '''
        solution_ind = solution > 1e-5 #True/False array
        svm_ind = np.arange(len(solution))[solution_ind]
        
        solution = solution[svm_ind]
        sv_x = X[svm_ind]
        sv_y = Y[svm_ind]
        f_sv_x=transform_kernel(sv_x)
        
        print("%d support vectors out of %d points" % (len(solution), n_samples))
        weights = np.zeros(n_features**2)
        for i,x in enumerate(f_sv_x):
            weights = weights+solution[i]*sv_y[i]*x
        
        print(weights)
        '''
        for i,x in enumerate(f_sv_x):
            b=sv_y[i]*np.dot(x,weights)
            print(b)
        '''
    
        '''Calculate b
        '''
        result=np.dot(f_sv_x,weights)*sv_y
        result_min_pos=np.extract(result>0, result).min()
        result_max_neg=np.extract(result<0, result).max()
        b=(result_min_pos+result_max_neg)/2
        print(b)
        #f_X = transform_kernel(X)
        #for i,
            
        return solution,weights,b
    
def transform_kernel(sv_x):
    ''' The function is to replicate transformation for polynomial kernel of degree 2 to the phi function . (c=0)
    '''
    n_samples, n_features = sv_x.shape
    f_sv_x=np.zeros((n_features**2)*n_samples).reshape(n_samples,n_features**2)
    for i,x_i in enumerate(sv_x):
        f_sv_x_i=[]
        for j,x_i_j in enumerate(x_i):
            for k,x_i_k in enumerate(x_i):
                f_sv_x_i.append(x_i_j*x_i_k)
        f_sv_x[i]=np.array(f_sv_x_i)
    return f_sv_x

def linear_kernel(x,y):
     return np.inner(x, y)

def polykernel(x,y,dimension, offset):
        return (offset + np.dot(x, y)) ** dimension

def gram_matrix(X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                #K[i, j] = linear_kernel(x_i, x_j)
                K[i, j] = polykernel(x_i, x_j,2,0)
        return K

if __name__ == '__main__':
    train_df = np.genfromtxt('data/mystery.data',delimiter=',')
    X = train_df[:,0:4]
    Y = train_df[:,4]
    multipliers,weights,b = compute_multipliers(X,Y,0)

    
                
        
    