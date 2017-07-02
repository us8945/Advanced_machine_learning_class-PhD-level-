'''
Created on Oct 31, 2016

@author: uri
Implement Naive Bayes model for SPAM classification
'''
import numpy as np
import numpy
import math
from copy import deepcopy
from numpy import std
import scipy.stats
import time

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
    
    #matrix_w_square=np.square(matrix_w)
    pi_distribution=np.mean(np.square(matrix_w),axis=1)
    return matrix_w,pi_distribution

def normalize(X,i_mean,i_std):
    ''' Perform column wise normalization
    '''
    if i_mean is None:
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
    else:
        mean=i_mean
        std=i_std
    
    X_norm=(X-mean)/std
    return X_norm, mean,std

def predict_summary(X,Y,gausian_params,prob_dictionary,ftr_indexes=None):
    #t=time.clock()
    correct=0
    missed=0
    prob_1=gausian_params[2][0]/(gausian_params[2][0]+gausian_params[2][1])
    prob_m1=gausian_params[2][1]/(gausian_params[2][0]+gausian_params[2][1])
    for i,rec in enumerate(X):
        label=predict_outcome(rec,gausian_params,prob_1,prob_m1,prob_dictionary,ftr_indexes)
        if label==Y[i]:
            correct+=1
        else:
            missed+=1
    
    #print ("Time to predict",(time.clock()-t))
    return correct,missed

def map_gausian_params(X,Y):
    X_1=X[np.where(Y==1)]
    X_m1=X[np.where(Y==-1)]
    mean_1 = np.mean(X_1,axis=0)
    std_1 = np.std(X_1,axis=0)
    mean_m1 = np.mean(X_m1,axis=0)
    std_m1 = np.std(X_m1,axis=0)
    return ((mean_1,std_1),(mean_m1,std_m1),(X_1.shape[0],X_m1.shape[0]))

def map_gausian_pdf_to_data_set(X_data,gausian_params):
    prob_dictionary={}
    for x in X_data:
        for col_ix in range(len(x)):
            col=x[col_ix]
            std_1=gausian_params[0][1][col_ix]
            mean_1=gausian_params[0][0][col_ix]
            std_m1=gausian_params[1][1][col_ix]
            mean_m1=gausian_params[1][0][col_ix]
            prob_1=scipy.stats.norm(mean_1,std_1).pdf(col)
            prob_m1=scipy.stats.norm(mean_m1,std_m1).pdf(col)
            if (col_ix,col,"one") in prob_dictionary.keys():
                continue
            else:
                prob_dictionary[(col_ix,col,"one")] = prob_1
                prob_dictionary[(col_ix,col,"minus_one")] = prob_m1
    return prob_dictionary

def predict_outcome(x,gausian_params,i_prob_1,i_prob_m1,prob_dictionary,ftr_indexes=None):
    #t=time.clock()
    if ftr_indexes is None:
        ftr_indexes=range(len(x))
    #print("Predict record", ftr_indexes)
    prob_1=i_prob_1
    prob_m1=i_prob_m1
    for col_ix in ftr_indexes:
        col=x[col_ix]
        std_1=gausian_params[0][1][col_ix]
        mean_1=gausian_params[0][0][col_ix]
        std_m1=gausian_params[1][1][col_ix]
        mean_m1=gausian_params[1][0][col_ix]
        prob_1=prob_1*prob_dictionary[(col_ix,col,"one")]
        prob_m1=prob_m1*prob_dictionary[(col_ix,col,"minus_one")]
        #prob_1=prob_1*scipy.stats.norm(mean_1,std_1).pdf(col)
        #prob_m1=prob_m1*scipy.stats.norm(mean_m1,std_m1).pdf(col)

    if (prob_1>prob_m1):
        return 1
    else:
        return -1
    
def map_feature_outcome(X,Y):
    '''Returns feature-map of the data set
       Feature-map has dictionary for each column in the column's order in the feature-map array
       The dictionary contains frequency of column value/label combination
       Last item of feature map has label frequency dictionary
    '''
    feature_map=[]
    for col in range(len(X[0])):
        feature_map.append({})
    feature_map.append({-1:0,1:0}) #add key for each label in last place
    
    for i,rec in enumerate(X):
        feature_map[len(X[0])][Y[i]]+=1 #add one to the record label counter
        for col in range(len(X[0])):
            if (rec[col],Y[i]) in feature_map[col].keys():
                feature_map[col][(round(rec[col],4),Y[i])]+=1
            else:
                feature_map[col][(round(rec[col],4),Y[i])]=1
        
    return feature_map

def predict_prob(feature_map,x,label_1_count,label_m1_count):
    result_dictionary={}
    prob_l1=label_1_count/(label_1_count+label_m1_count)
    prob_lm1=label_m1_count/(label_1_count+label_m1_count)
    for i,col in enumerate(x):
        prob_l1=prob_l1*(feature_map[i].get((round(col,4),1),0.1)/label_1_count)
        prob_lm1=prob_lm1*(feature_map[i].get((round(col,4),-1),0.1)/label_m1_count)
        #print(i,col,prob_l1,prob_lm1,feature_map[i].get((round(col,4),1),0),feature_map[i].get((round(col,4),-1),0))
    
    return prob_l1,prob_lm1

def load_and_normalize_data():
    train_df = np.genfromtxt('data/spam_train.data',delimiter=',',dtype=np.longdouble)
    #train_df = train_df[0:500,:] #use for testing
    test_df = np.genfromtxt('data/spam_test.data',delimiter=',',dtype=np.longdouble)
    validation_df = np.genfromtxt('data/spam_validation.data',delimiter=',',dtype=np.longdouble)
    n_samples, n_features = train_df.shape
    X_train = train_df[:,0:n_features-1]
    Y_train = train_df[:,n_features-1]
    Y_train[np.where(Y_train==0)]=-1 #replace zero with -1
    
    n_samples, n_features = validation_df.shape
    X_validation = validation_df[:,0:n_features-1]
    Y_validation = validation_df[:,n_features-1]
    Y_validation[np.where(Y_validation==0)]=-1 #replace zero with -1
    
    n_samples, n_features = test_df.shape
    X_test = test_df[:,0:n_features-1]
    Y_test = test_df[:,n_features-1]
    Y_test[np.where(Y_test==0)]=-1 #replace zero with -1
    
    X_train,x_mean,x_std = normalize(X_train, None, None)
    X_validation,x_mean,x_std=normalize(X_validation, x_mean, x_std)
    X_test,x_mean,x_std=normalize(X_test, x_mean, x_std)
    
    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

if __name__ == '__main__':
    X_train,Y_train,X_validation,Y_validation,X_test,Y_test=load_and_normalize_data()
    gausian_params = map_gausian_params(X_train,Y_train)
    test_prob_dictionary=map_gausian_pdf_to_data_set(X_test,gausian_params)
    train_prob_dictionary=map_gausian_pdf_to_data_set(X_train,gausian_params)
    validation_prob_dictionary=map_gausian_pdf_to_data_set(X_validation,gausian_params)
    
    correct,missed = predict_summary(X_train,Y_train,gausian_params,train_prob_dictionary)
    print("Training accuracy:",(correct/(correct+missed)))
    
    correct,missed = predict_summary(X_test,Y_test,gausian_params,test_prob_dictionary)
    print("Testing accuracy:",(correct/(correct+missed)))
    
    correct,missed = predict_summary(X_validation,Y_validation,gausian_params,validation_prob_dictionary)
    print("Validation accuracy:",(correct/(correct+missed)))
    
    #Training accuracy: 0.898
    #Validation accuracy: 0.77625
    #Testing accuracy: 0.6179775280898876
    
    pca_vec=[1,2,3,4,5,6,7,8,9,10]
    #pca_vec=[10]
    for num_vec in pca_vec:
        W,pi_dist=calc_pca(X_train, num_vec)
        #print(pi_dist)
        s=range(1,21) #to generate values from 1 to 20
        #s=[20]
        for num_features in s:
            accuracy=0
            best_accuracy=0
            for i in range(100): #perform 100 iterations of selecting features and checking accuracy
                ftr_indexes=np.random.choice(a=len(pi_dist), size=num_features, replace=True, p=pi_dist)
                ftr_indexes=np.unique(ftr_indexes)
                #print(ftr_indexes)
                gausian_params = map_gausian_params(X_train,Y_train)
                correct,missed = predict_summary(X_test,Y_test,gausian_params,test_prob_dictionary,ftr_indexes)
                if best_accuracy<(correct/(correct+missed)):
                    best_accuracy=(correct/(correct+missed))
                    best_features=ftr_indexes
                accuracy+=(correct/(correct+missed))
            
            correct,missed = predict_summary(X_validation,Y_validation,gausian_params,validation_prob_dictionary,best_features)
            print("Testing, k-dim;",num_vec," s-features:",num_features," average accuracy:",round(accuracy/100,4)," best accuracy:",round(best_accuracy,4),"best features:",best_features,"Validation accuracy on best features:",(correct/(correct+missed)))
#         x_train_transform=X_train.dot(W)
#         x_test_transform=X_test.dot(W)
#         x_validation_transform=X_validation.dot(W)
#         multipliers=compute_primal_with_slack(x_train_transform,Y_train,c)
#         multipliers = multipliers[0:num_vec+1] #filter out slack multipliers
#         np.set_printoptions(threshold=np.inf)
#         correct,missed = predict_summary(x_train_transform,Y_train,multipliers)
#         print("Training set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
#         correct,missed = predict_summary(x_validation_transform,Y_validation,multipliers)
#         print("Validation set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
#         correct,missed = predict_summary(x_test_transform,Y_test,multipliers)
#         print("Testing set slack and k-dim:",c,num_vec,"accuracy:",(correct/(correct+missed)))
    
    
                
        
    