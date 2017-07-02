'''
Created on Sep 25, 2016

@author: uri
Implement decision trees using the information gain heuristic to classify heart data set
Data from:
https://archive.ics.uci.edu/ml/datasets/SPECT+Heart
'''
import csv
import math


class d_node(object):
    def __init__(self,column=None,value=None,true_node=None,false_node=None,classification=None):
        self.column = column
        self.value = value
        self.true_node=true_node
        self.false_node=false_node
        self.classification=classification #Holds dictionary in case of leaf node
        
class decision_tree(object):
    def __init__(self,train):
        self.train_data = train
        self.head = None
        
    def get_num_col_features(self,data_set):
        columns_num = len(data_set[0])
        num_columns=0
        num_features=0
        for col in range(1,columns_num):
            num_columns=num_columns+1
            values = set(rec[col] for rec in train) #get set of unique values in column "col"
            num_features=num_features+len(values)
            #print("Column,features",col,len(values))

        print("Total columns, features",num_columns,num_features)
    
    def best_single_node(self):
        columns_num = len(train[0])
        best_accuracy=0.0
        for col in range(1,columns_num):
            values = set(rec[col] for rec in train) #get set of unique values in column "col"
            for val in values:
                set1,set2=self.split_data_set(self.train_data, col, val)
                labels1=self.get_uniques(set1)
                labels2=self.get_uniques(set2)
                #labels1 = defaultdict(lambda: 0, labels1)
                #labels2 = defaultdict(lambda: 0, labels2)
                #key_p=labels1['p']+labels2['p']
                if labels1 == {}:
                    key1_max=0
                else:
                    key1_max=labels1[max(labels1, key=labels1.get)]
                
                if labels2 =={}:
                    key2_max=0
                else:
                    key2_max=labels2[max(labels2, key=labels2.get)]
                    
                accuracy = (key1_max+key2_max)/len(self.train_data)
                if accuracy > best_accuracy:
                    best_accuracy=accuracy
                    best_set1=set1
                    best_set2=set2
                    best_column=col
                    best_value=val
        print("Best one node split",best_accuracy,best_column,best_value)
        return best_accuracy,best_column,best_value
        
    def fit_model(self):
        self.head=self._fit_model(self.train_data)
    
    def print_tree(self,node=None,print_ident='',print_level=0):
        if node is None:
            node=self.head #Start from the top of the tree
        if node.classification is not None:
            print(print_ident+'   '+str(node.classification)) # We reached leaf node
        else:
            print(print_ident,"Column Sequence",node.column,",value:",node.value)
            print_ident=print_ident+'   '
            print_level+=1
            print(str(print_level)+print_ident+"True--->")
            self.print_tree(node.true_node, print_ident,print_level)
            print(str(print_level)+print_ident+"False--->")
            self.print_tree(node.false_node, print_ident,print_level)
    
    def predict(self,rec,node=None):
        if node is None: #we are in the root of the tree
            node=self.head
        if node.classification is not None:
            result=max(node.classification, key=node.classification.get)
            #print('Found leaf',node.classification,result)
            return  result       #return classification with maximum votes
        else:
            if rec[node.column]==node.value:
                return self.predict(rec, node.true_node)
            else:
                return self.predict(rec, node.false_node)
    
    def head_weights(self):
        true_labels=[]
        false_labels=[]
        self.get_branch_weights(self.head.true_node,true_labels)
        self.get_branch_weights(self.head.false_node,false_labels)
        print("True branch labels",true_labels)
        print("False branch labels",false_labels)
        true_lab_d={}
        false_lab_d={}
        for d in true_labels:
            for key in d:
                if key in true_lab_d.keys():
                    true_lab_d[key]=true_lab_d[key]+d[key]
                else:
                    true_lab_d[key]=d[key]
        
        for d in false_labels:
            for key in d:
                if key in false_lab_d.keys():
                    false_lab_d[key]=false_lab_d[key]+d[key]
                else:
                    false_lab_d[key]=d[key]
        
        print("True branch summary",true_lab_d)
        print("False branch summary",false_lab_d)
    
    def get_branch_weights(self,node,labels):
        if node.classification is not None:
            labels.append(node.classification)       #return classification
        else:
                self.get_branch_weights(node.true_node,labels)
                self.get_branch_weights(node.false_node,labels)
        
    def _fit_model(self,train):
        if len(train)==0:
            return d_node()
        
        current_entropy = self.get_entropy(train)
        columns_num = len(train[0])
        best_gain=0.0
        for col in range(1,columns_num):
            values = set(rec[col] for rec in train) #get set of unique values in column "col"
            
            for val in values:
                set1,set2=self.split_data_set(train, col, val)
                split_ratio = len(set1)/len(train)
                new_entropy = split_ratio*self.get_entropy(set1) + (1-split_ratio)*self.get_entropy(set2)
                information_gain = current_entropy - new_entropy
                if information_gain > best_gain:
                    best_gain=information_gain
                    best_set1=set1
                    best_set2=set2
                    best_column=col
                    best_value=val
        if best_gain>0:
            #print(current_entropy,best_gain,best_column,best_value,len(train),len(best_set1),len(best_set2))
            true_branch=self._fit_model(best_set1)
            false_branch=self._fit_model(best_set2)
            return d_node(column=best_column,value=best_value,true_node=true_branch,false_node=false_branch)
        else:
            #print(self.get_uniques(train))
            return d_node(classification=self.get_uniques(train)) #Reached leaf node
        
    def get_uniques(self,train):
        result={}
        for rec in train:
            if rec[0] in result.keys():
                result[rec[0]] = result[rec[0]]+1
            else:
                result[rec[0]] = 1
        return result
    
    def split_data_set(self,train,col,value):
        set1=[]
        set2=[]
        for t in train:
            if t[col]==value:
                set1.append(t)
            else:
                set2.append(t)
        
        return set1,set2
    
    
    def get_entropy(self,train):
        ''' Calculate entropy:
              - First get dictionary of labels with value indicating frequency
              - Calculate entropy by summing probabilities and its logs:  -1(Prob*Log2(Prob))
        '''
        results = self.get_uniques(train)
        total_rec=len(train)
        entropy=0.0
        for r in results.keys():
            p=float(results[r]/total_rec)
            entropy = entropy - p*math.log2(p)
        return entropy

if __name__ == '__main__':
    f=open('data/heart_train.data')
    reader = csv.reader(f, delimiter=',')
    train=[]
    for rec in reader:
        train.append(rec)
    f.close()
    f=open('data/heart_test.data')
    reader = csv.reader(f, delimiter=',')
    test=[]
    for rec in reader:
        test.append(rec)
    
    decision_tr=decision_tree(train)
    decision_tr.fit_model()
    decision_tr.print_tree()
    decision_tr.get_num_col_features(train)
    
    correct=0
    incorrect=0
    for rec in train:
        result=decision_tr.predict(rec)
        if result==rec[0]:
            correct=correct+1
        else:
            incorrect=incorrect+1
    
    print("Training Accuracy:",correct/(correct+incorrect))
    
    correct=0
    incorrect=0
    for rec in test:
        result=decision_tr.predict(rec)
        #print(result,rec[0])
        if result==rec[0]:
            correct=correct+1
        else:
            incorrect=incorrect+1
    
    print("Testing Accuracy:",correct/(correct+incorrect))
        
    
    
                
        
    