'''
Created on Nov 26, 2016

@author: Uri Smashnov
Bayesian Networks
Use zoo.data data set provided with this problem set. This data
set was generated from the UCI Zoo data set by removing the animal names: https://archive.ics.uci.edu/ml/datasets/Zoo
Label is the last column in the data set.
Learn a discriminative model for the class label by using the Chow-Liu Bayesian structure.


Acknowledgements: 
    Logistic regression part only: consulted from https://gist.github.com/yusugomori/4462221
    convertToOneHot function taken from http://stackoverflow.com/questions/29831489/numpy-1-hot-array
Flow:
- Read input data
- Populate graph nodes
- Connect every node with every node using Edge
- Assign weight to each Edge, which mutual information between two nodes. Multiply by (-1) since MAX is not available, only MIN
- Use complete Graph above to calculate Min-spanning tree
- Use Spanning tree for prediction model:
    - Every node connected to Label node is important
    - Every node not connected to Label node is not used
    - Use Multi-Class Logistic Regression to Train the model
'''

import numpy as np
import networkx as nx 
import math
import matplotlib.pyplot as plt
#import pydotplus
#import graphviz
np.seterr(all='ignore')

def convertToOneHot(vector, num_classes=None):
    """
    http://stackoverflow.com/questions/29831489/numpy-1-hot-array
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)
    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = np.zeros((n_in, n_out))  
        self.b = np.zeros(n_out)          


    def train(self, lr=0.1, rounds=100): 
        for n in range(rounds):
            p_y_given_x = softmax(np.dot(self.x, self.W) + self.b)
            d_y = self.y - p_y_given_x
            
            self.W += lr * np.dot(self.x.T, d_y) # - lr * L2_reg * self.W
            self.b += lr * np.mean(d_y, axis=0)


    def predict(self, x):
        return softmax(np.dot(x, self.W) + self.b)


def calculate_mulual_information(data,i,j):
    mutual_info = 0.0
    n_samples, n_features = data.shape
    res_dic={}
    res_i_dic={}
    res_j_dic={}
    for rec in data:
        if ((rec[i],rec[j]) in res_dic.keys()):
            res_dic[(rec[i],rec[j])] += 1
            res_i_dic[rec[i]] +=1
            res_j_dic[rec[j]] +=1 
        else:
            res_dic[(rec[i],rec[j])] = 1
            if rec[i] in res_i_dic.keys():
                res_i_dic[rec[i]] +=1
            else:
                res_i_dic[rec[i]] =1
            
            if rec[j] in res_j_dic.keys():
                res_j_dic[rec[j]] +=1
            else:
                res_j_dic[rec[j]] =1

    for key in res_dic.keys():
        mutual_info = mutual_info + (res_dic[key]/n_samples)*math.log2((res_dic[key]*n_samples)/(res_i_dic[key[0]]*res_j_dic[key[1]]))
    
    return mutual_info



zoo_df = np.genfromtxt('data/zoo.data',delimiter=',',dtype=np.int)
n_samples, n_features = zoo_df.shape

G=nx.Graph()
for i in range(n_features):
    G.add_node(i)

for i in range(n_features):
    for j in range(n_features):
        if (i!=j):
            mut_info=calculate_mulual_information(zoo_df, i, j)
            G.add_edge(i,j,weight=(-1)*mut_info) #Reverse mutual info since we will use minimum instead of maximum
            #G.add_edge(i,j,weight=(-1)*mut_info)


T_mx_spanning = nx.minimum_spanning_tree(G)


# Show and prompt to save picture
nx.draw_networkx(T_mx_spanning)
plt.draw()

columns=[] #Identify columns directly connected to Label column
for edge in T_mx_spanning.edges():
    if edge[0] == (n_features-1):
        columns.append(edge[1])
    elif edge[1] == (n_features-1):
        columns.append(edge[0])

'''Construct X matrix only from relevant columns, connected directly to the Label column in the Tree
'''        
X_zoo = zoo_df[:,columns]
Y_zoo = zoo_df[:,n_features-1]
X_train = zoo_df[:,0:n_features-1]
''' Encode Y into binary matrix with one column for each potential value of the Label. Each record will have 1 only in one column
'''
Y_binary = convertToOneHot(Y_zoo)
classifier = LogisticRegression(input=X_zoo, label=Y_binary, n_in=X_zoo.shape[1], n_out=Y_binary.shape[1])
learning_rate=0.1
classifier.train(lr=learning_rate,rounds=50)
correct=0
for i in range(n_samples):
    if (np.argmax(classifier.predict(X_zoo[i]), axis=0) == Y_zoo[i]):
        correct+=1
        
print('Accuracy rate on training set is', round(correct/len(X_zoo),4))

''' Compute accuracy on Label "1" only
'''
correct=0
total_one=0
for i in range(n_samples):
    if (np.argmax(classifier.predict(X_zoo[i]), axis=0) == Y_zoo[i]) and (Y_zoo[i]== 1):
        correct+=1
    if Y_zoo[i]== 1:
        total_one+=1
        
print('Accuracy rate on "1" only (training set) is', round(correct/total_one,4), "Total number of labels one:",total_one)
plt.show()
