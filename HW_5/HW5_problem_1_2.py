'''
Created on Nov 26, 2016

@author: Uri Smashnov
    
Bayesian Networks
Use zoo.data data set provided with this problem set. This data
set was generated from the UCI Zoo data set by removing the animal names: https://archive.ics.uci.edu/ml/datasets/Zoo
Label is the last column in the data set.
Learn a discriminative model for the class label by using the Chow-Liu Bayesian structure.
Problem 1-2 - generate 20 random trees, learn parameters and run Coordinate descent to classify

Flow:
- Read input data
- Generate 20 Random trees:
    - Populate graph nodes
    - Connect every node with every node using Edge
    - Assign Random weight to each Edge
    - Use complete Graph above to calculate Min-spanning tree
    - Learn Tree parameters using MLE and Naieve Bayes assumprion: 
        - all nodes connected to the Label node are independent
        - all nodes that are not connected directly to Label node are not used 
- Calculate map of predictions for every modela and every training record
- Run Coordinate descent on Hypothesis space of 20 random trees generated above, each tree is input to Naive Bayes Prediction model
- Model details:
    - Every node connected to Label node is important
    - Every node not connected to Label node is not used 
- Use Alpha's output from above for Final prediction model by multiplying each model output by its alpha
- Run 100 iterations and calculate minimum, maximum and average prediction rate

'''

import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import pydotplus
import graphviz
import random
import operator

class CoordinateDescent(object):
    def __init__(self,train,Y,models):
        self.train = train
        self.Y = Y
        self.alpha=[]
        self.models=models

        ''' Initialize alpha: one divided by number of non-label columns
        '''
        #init_alpha=1/(len(models))
        init_alpha=0
        for i in range(len(models)):
            self.alpha.append(init_alpha)
        
    
    def fit(self,rounds=None):
        #print(self.alpha)
        exp_loss=float('inf')
                
        while(1):
        #for j in range(1):
            for i,alpha in enumerate(self.alpha):
                self.update_alpha(i)
                new_exp_loss=self.calc_exp_loss()
                #print("New exp_loss",new_exp_loss)
                if new_exp_loss<exp_loss:
                    exp_loss=new_exp_loss
                else:
                    print('Exp loss',exp_loss,', Alphas',self.alpha)
                    return
            
    
    def calc_exp_loss(self):
        exp_loss=0.0
        for rec_num in range(len(self.train)):
            exp_loss+=math.exp((-1)*self.Y[rec_num]*self.predict(rec_num))
        
        return exp_loss
        
    def update_alpha(self,index_alpha):
        correct_models=0
        incorrect_models=0
        exp_upper_coef=0.0
        exp_lower_coef=0.0
        for train_rec_num in range(len(self.train)):
            result = self.models[index_alpha][0][train_rec_num]
            if result == self.Y[train_rec_num]:
                correct_models+=self._predict_models(train_rec_num,index_alpha) #predict all models but index_alpha
                #print('Coorect models',correct_models,index_alpha,train_rec_num)
            else:
                incorrect_models+=self._predict_models(train_rec_num,index_alpha) #predict all models but index_alpha
                #print('Incorrect models',correct_models,index_alpha,train_rec_num)
        
        #print("Incorrect=",incorrect_models)
        #print("Correct=", correct_models)
        if incorrect_models==0:
            new_alpha=0
        else:
            new_alpha = 0.5*math.log(correct_models/incorrect_models)
        
        self.alpha[index_alpha]=new_alpha
    
    def _predict_models(self,train_rec_num,index_alpha):
        ''' Return summary of models, each model multiplied by its alpha
        ''' 
        results_total=0
        for i,model in enumerate(self.models):
            if i!=index_alpha:
                results_total+=model[0][train_rec_num]*self.alpha[i]
        
        return math.exp((-1)*self.Y[train_rec_num]*results_total)
        
    
        
    def predict(self,test_rec_num):
        ''' predict single record label by using all models 
            in the tree_dictionary, by combining their results and multiplying
            each by alpha of the model
        '''
        predicted_label = 0
        for i,model in enumerate(self.models):
            #print(tree.predict(test_rec)*tree.alpha)
            label=model[0][test_rec_num]
            predicted_label = predicted_label + label*self.alpha[i]
        
        return predicted_label


def map_label_column_probability(X,Y):
    '''Returns feature-map of the data set
       The dictionary has number of records per tupel: (Label, Column_number, Column_value)
    '''
    feature_map={}
    
    label_dict={}    
    for label in np.unique(Y):
        label_dict[label]=0
    
    for i,rec in enumerate(X):
        label_dict[Y[i]]+=1                   #add one to the current label in Label dictionary
        for col in range(len(X[0])):
            if (Y[i],col,rec[col]) in feature_map.keys():
                feature_map[(Y[i],col,rec[col])]+=1
            else:
                feature_map[(Y[i],col,rec[col])]=1
    
    #print('Ftr map',feature_map)
    return feature_map, label_dict


def predict_single_record(rec,columns_ind,ftr_map,label_dict,num_records):
    
    prob_1=label_dict[1]/num_records #Set to probability of the Label in the data set
    prob_m1=label_dict[-1]/num_records
    #print('Initial prob',labels[label-1])
    for col in columns_ind:
        prob_1 =prob_1*(ftr_map.get((1,col,rec[col]),0)/label_dict[1])
        prob_m1 =prob_m1*(ftr_map.get((-1,col,rec[col]),0)/label_dict[-1])
        
    if prob_1>prob_m1:
        return 1
    else:
        return -1 
    
def predict_labels(X,Y,columns_ind,ftr_map,label_dict,num_records):
    predict_labels=[]
    correct=0
    incorrect=0
    for i,rec in enumerate(X):
        label = predict_single_record(rec,columns_ind,ftr_map,label_dict,num_records)
        predict_labels.append(label)
        if label==Y[i]:
            correct+=1
        else:
            incorrect+=1
    
    return predict_labels,correct, incorrect
    
'''Load data set, calculate probabilities, generate random models and ran Coordinate descent
'''
zoo_df = np.genfromtxt('data/zoo.data',delimiter=',',dtype=np.int)
n_samples, n_features = zoo_df.shape
X_zoo = zoo_df[:,0:n_features-1]
Y_zoo = zoo_df[:,n_features-1]
Y_zoo[np.where(Y_zoo!=1)]=-1    #replace Labels not equal to 1 with -1

feature_map, label_dict = map_label_column_probability(X_zoo,Y_zoo)
#print(X_zoo[2],Y_zoo)
#print(predict_single_record(X_zoo[2],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],feature_map,label_dict,len(Y_zoo)))
#print(predict_labels(X_zoo,Y_zoo,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],feature_map,label_dict,len(Y_zoo)))

accuracy_arr=[]
for j in range(100):
    models=[]
    for r in range(20):
        G=nx.Graph()
        for i in range(n_features):
            G.add_node(i)
        
        for i in range(n_features):
            for j in range(n_features):
                if (i!=j):
                    '''Randomly assign weight to generate random spanning tree
                    '''
                    G.add_edge(i,j,weight=random.random()) 
        
        ''' Identify min spanning tree (random because weights are random), and learn NB model
        '''
        T_mx_spanning = nx.minimum_spanning_tree(G)
        columns=[] #Identify columns directly connected to Label column
        for edge in T_mx_spanning.edges():
            if edge[0] == (n_features-1):
                columns.append(edge[1])
            elif edge[1] == (n_features-1):
                columns.append(edge[0])
        
        models.append(predict_labels(X_zoo,Y_zoo,columns,feature_map,label_dict,len(Y_zoo)))  
    
    #for model in models:
    #    print(model,model[1]/len(Y_zoo))  
    
    classifier=CoordinateDescent(X_zoo,Y_zoo,models)
    classifier.fit()
    correct=0
    incorrect=0
    for train_rec_num in range(len(X_zoo)):
            result = classifier.predict(train_rec_num)
            if (Y_zoo[train_rec_num]==1 and result>0) or (Y_zoo[train_rec_num]==-1 and result<0):
                correct+=1
            else:
                incorrect+=1
            #print(train_rec[1],'Label',train_rec[0],'Result',result)
    print("Training Accuracy:",correct/(correct+incorrect))
    accuracy_arr.append(round(correct/(correct+incorrect),2))

print("Accuracy min:",min(accuracy_arr), ', Accuracy max:',max(accuracy_arr), 'Accuracy Average:',sum(accuracy_arr)/len(accuracy_arr))
    
