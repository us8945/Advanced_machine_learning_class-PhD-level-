'''
Created on Sep 25, 2016

@author: uri
Assume hypothesis space consists only of depth 2 decision trees only.

Run the adaBoost algorithm with M = 4 to train a classifier for this data set.
Use different values of M.

'''
import csv
import math
import copy

class d_node(object):
    def __init__(self,column=None,value=None,true_node=None,false_node=None,classification=None):
        self.column = column
        self.value = value
        self.true_node=true_node
        self.false_node=false_node
        self.classification=classification #Holds dictionary in case of leaf node
        
class decision_tree(object):
    def __init__(self,train,max_depth=None,weights=[]):
        self._copy_data(train) # copy data to self.train_data
        self.len_train = len(train)
        self.weights = weights
        self.head = None
        self.max_depth = max_depth
        self.current_depth=0
        self.alpha=0
        ''' Place weights as last column of the train matrix
        '''
        self._place_weights()
    
    def _copy_data(self,train):
        self.train_data=[]
        for rec in train:
            new_rec=rec[:]
            self.train_data.append(new_rec)
            
    def set_tree_parameters(self,alpha,error):
        self.alpha=alpha
        self.error=error
    
    def _place_weights(self):
        ''' Add weight as last column in the matrix
            If weights vector in None, add equal weights
        '''
        if self.weights == []:
            for i in range(len(self.train_data)):
                self.train_data[i].append(float(1/len(self.train_data)))
                self.weights.append(float(1/len(self.train_data)))
        else:
            for i in range(len(self.train_data)):
                self.train_data[i].append(self.weights[i])
            
    def fit_model_greedy(self):
        columns_num = len(self.train_data[0])-1 #Last column is weights
        #print("Num columns",columns_num,len(train[0]))
        best_error=1.0
        for col1 in range(1,columns_num):
            value = '0'
            set1,set2=self.split_data_set(self.train_data, col1, value)
            for col2 in range(1,columns_num):
                if col1==col2:
                    continue
                else:
                    set21,set22=self.split_data_set(set1, col2, value)
                    left_true_node =  d_node(classification=self.get_uniques(set21))
                    left_false_node = d_node(classification=self.get_uniques(set22))
                    left_node= d_node(column=col2,value='0',true_node=left_true_node,false_node=left_false_node)
                    for col3 in range(1,columns_num):
                        if col1==col3 or col2==col3:
                            continue
                        else:
                            set31,set32=self.split_data_set(set2, col3, value)
                            right_true_node =  d_node(classification=self.get_uniques(set31))
                            right_false_node = d_node(classification=self.get_uniques(set32))
                            right_node= d_node(column=col3,value='0',true_node=right_true_node,false_node=right_false_node)
            
                            head_node = d_node(column=col1,value='0',true_node=left_node,false_node=right_node)
                            error = self.predict_bulk(self.train_data, head_node)
                            if error < best_error:
                                best_error = error
                                self.head = head_node
                        
        #print("Best error",best_error)
        return best_error
        
    def fit_model_using_entropy(self):
        '''Fit model using information gain
        '''
        self.head=self._fit_model_entropy(self.train_data,self.max_depth)
    
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
    
    def predict_bulk(self,prd_data,node=None):
        error=0.0
        for i,rec in enumerate(prd_data):
            label,weight = self.predict(rec, node)
            if (label>0 and rec[0]>0) or (label<0 and rec[0]<0):
                continue
            else:
                error=error+self.weights[i]
        
        return error 
            
            
    def predict(self,rec,node=None):
        if node is None: #we are in the root of the tree
            node=self.head
        if node.classification is not None:
            result=max(node.classification, key=node.classification.get)
            weight=node.classification[result]
            #print('Found leaf',node.classification,result)
            return  result,weight       #return classification with maximum votes
        else:
            #print(node.column)
            if rec[node.column]==node.value:
                return self.predict(rec, node.true_node)
            else:
                return self.predict(rec, node.false_node)
        
    def _fit_model_entropy(self,train,depth):
        cur_depth = depth
        if len(train)==0:
            return d_node()
        
        #print("Max-depth,current-depth",self.max_depth,self.current_depth)
        if self.max_depth+1 < cur_depth:
            #print("Reached maximum depth")
            return d_node(classification=self.get_uniques(train))
        
        cur_depth+=1
        current_entropy = self.get_entropy(train)
        columns_num = len(train[0])-1 #Last column is weights
        #print("Num columns",columns_num,len(train[0]))
        best_gain=0.0
        for col in range(1,columns_num):
            values = set(rec[col] for rec in train) #get set of unique values in column "col"
            
            for val in values:
                set1,set2=self.split_data_set(train, col, val)
                split_ratio = self.get_sum_weights(set1)/self.get_sum_weights(train)
                new_entropy = split_ratio*self.get_entropy(set1) + (1-split_ratio)*self.get_entropy(set2)
                #print("Column,value,entropy")
                information_gain = current_entropy - new_entropy
                if information_gain > best_gain:
                    best_gain=information_gain
                    best_set1=set1
                    best_set2=set2
                    best_column=col
                    best_value=val
        if best_gain>0:
            #print("Best gain",current_entropy,best_gain,best_column,best_value,len(train),len(best_set1),len(best_set2))
            true_branch=self._fit_model_entropy(best_set1,cur_depth)
            false_branch=self._fit_model_entropy(best_set2,cur_depth)
            return d_node(column=best_column,value=best_value,true_node=true_branch,false_node=false_branch)
        else:
            #print(self.get_uniques(train))
            return d_node(classification=self.get_uniques(train)) #Reached leaf node
        
    def get_sum_weights(self,train):
        if train==[]:
            return 0.0
        weight_ind=len(train[0])-1
        train_weight=0.0
        for r in train:
            train_weight=train_weight+r[weight_ind]
        
        return train_weight
                
                   
    def get_uniques(self,train):
        result={}
        if train==[]:
            return result
        
        weight_ind=len(train[0])-1
        for rec in train:
            if rec[0] in result.keys():
                result[rec[0]] = result[rec[0]]+rec[weight_ind]
            else:
                result[rec[0]] = float(rec[weight_ind])
        
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
        train_weight=self.get_sum_weights(train)
        entropy=0.0
        for r in results.keys():
            p=float(results[r]/train_weight)
            #p=float(results[r]/total_rec)
            #print(entropy,p,r,results[r],total_rec)
            entropy = entropy - p*math.log2(p)
            
        return entropy

class AdaBoost(object):
    def __init__(self,train,M=4,tree_depth=2):
        self.train = train
        self.M = M
        self.tree_depth=tree_depth
    
    def predict(self,test_rec):
        ''' predict single record label by using all models 
            in the tree_dictionary, by combining their results and multipling
            each by alpha of the model
        '''
        predicted_label = 0
        for tree in self.trees_dic.values():
            #print(tree.predict(test_rec)*tree.alpha)
            label,weight=tree.predict(test_rec)
            predicted_label = predicted_label + label*tree.alpha
        
        return predicted_label
    
    def fit_greedy(self):
        error=0.0
        weights=[]
        self.trees_dic={}
        for i in range(len(self.train)):
            weights.append(float(1/len(self.train)))
                
        for i in range(self.M):
            #print("Data coluns len",len(self.train[0]))
            decision_tr=decision_tree(self.train,self.tree_depth,weights)
            #print("Data coluns len 2",len(self.train[0]))
            decision_tr.fit_model_greedy()
            #print("Data coluns len 3",len(self.train[0]))
            error,results,labels=self._predict_single_tree(decision_tr, self.train)
            #print(error,results,labels)
            alpha=0.5*math.log((1-error)/error)
            weights = self.update_weights(weights,alpha,error,results,labels)
            decision_tr.set_tree_parameters(alpha,error)
            self.trees_dic[i] = decision_tr
        
        
    def fit_entropy(self):
        error=0.0
        weights=[]
        self.trees_dic={}
        for i in range(len(self.train)):
            weights.append(float(1/len(self.train)))
                
        for i in range(self.M):
            #print("Data coluns len",len(self.train[0]))
            decision_tr=decision_tree(self.train,self.tree_depth,weights)
            #print("Data coluns len 2",len(self.train[0]))
            decision_tr.fit_model_using_entropy()
            #print("Data coluns len 3",len(self.train[0]))
            error,results,labels=self._predict_single_tree(decision_tr, self.train)
            alpha=0.5*math.log((1-error)/error)
            weights = self.update_weights(weights,alpha,error,results,labels)
            decision_tr.set_tree_parameters(alpha,error)
            self.trees_dic[i] = decision_tr
        
        #print(self.trees_dic)
    
    def update_weights(self,weights,alpha,error,results,labels):
        n_weights=[]
        new_alpha=math.fabs(alpha)
        for i,w in enumerate(weights):
            n_weights.append(w*((math.exp((-1)*labels[i]*results[i]*new_alpha))/(2*math.sqrt(error*(1-error)))))
        
        '''Make sum of weights equal to 1, to take care of calculation error
        '''
        total_weights=0
        for w in n_weights:
            total_weights=total_weights+w
        for i,w in enumerate(n_weights):
            n_weights[i] = w/total_weights
            
        #print("New Weights",n_weights)
        return n_weights
        
    def _predict_single_tree(self,d_tree,pred_data):     
        correct=0
        incorrect=0
        results=[]
        labels=[]
        error=0.0
        for i,record in enumerate(pred_data):
            result,weight=d_tree.predict(record)
            results.append(result)
            labels.append(record[0])
            if result==record[0]:
                correct=correct+1
            else:
                incorrect=incorrect+1
                error=error+d_tree.weights[i]
        
        return error,results,labels

def read_data(file_name):
    f=open(file_name)
    reader = csv.reader(f, delimiter=',')
    data=[]
    for rec in reader:
        if rec[0]=='0':
            rec[0]=-1
        else:
            rec[0]=1
        data.append(rec)
    f.close()
    return data            

if __name__ == '__main__':
    train = read_data('data/heart_train.data')
    test = read_data('data/heart_test.data')
    
    model=AdaBoost(train,M=4,tree_depth=2)
    model.fit_greedy()
    for i in range(4):
        print("Tree number:",i+1)
        print("Tree error:",model.trees_dic[i].error)
        print("Tree alpha:",model.trees_dic[i].alpha)
        model.trees_dic[i].print_tree()
        
    
    correct=0
    incorrect=0
    for rec in train:
        result=model.predict(rec)
        #print("Result, Label",result,rec[0])
        if (result>0 and rec[0]>0) or (result<0 and rec[0]<0):
            correct=correct+1
        else:
            incorrect=incorrect+1
    
    print("Training Accuracy:",correct/(correct+incorrect))
    
    correct=0
    incorrect=0
    for rec in test:
        result=model.predict(rec)
        #print("Result, Label",result,rec[0])
        if (result>0 and rec[0]>0) or (result<0 and rec[0]<0):
            correct=correct+1
        else:
            incorrect=incorrect+1
    
    print("Testing Accuracy:",correct/(correct+incorrect))
    

    for m in [8,16,32,64]:
        model=AdaBoost(train,M=m,tree_depth=2)
        model.fit_greedy()
        correct=0
        incorrect=0
        for rec in train:
            result=model.predict(rec)
            #print("Result, Label",result,rec[0])
            if (result>0 and rec[0]>0) or (result<0 and rec[0]<0):
                correct=correct+1
            else:
                incorrect=incorrect+1
        
        print("Training Accuracy,m:",m,correct/(correct+incorrect))
        
        correct=0
        incorrect=0
        for rec in test:
            result=model.predict(rec)
            #print("Result, Label",result,rec[0])
            if (result>0 and rec[0]>0) or (result<0 and rec[0]<0):
                correct=correct+1
            else:
                incorrect=incorrect+1
        
        print("Testing Accuracy,m:",m,correct/(correct+incorrect))   
    
    
                
        
    