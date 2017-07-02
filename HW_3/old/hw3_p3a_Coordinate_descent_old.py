'''
Created on Oct 19, 2016

@author: uri
'''
import csv
import math

class d_node(object):
    def __init__(self,column=None,value=None,true_node=None,false_node=None,classification=None):
        self.column = column
        self.value = value
        self.true_node = true_node
        self.false_node = false_node
        self.classification=classification #Holds dictionary in case of leaf node

class OneLevelTree(object):
    def __init__(self,train,column):
        self.train = train
        self.head=d_node(column,'1',None,None,None)
        self.column=column
    
    def print_tree(self):
        print('Column:',self.column,'val=1',self.head.true_node.classification,'val=0',self.head.false_node.classification)
        #print('Column:',self.column,'=0',self.head.false_node.classification)
        
    def predict(self,rec):
        #print(self.head.true_node.column,rec[self.head.true_node.column],self.head.true_node.value)
        if rec[self.head.true_node.column]==self.head.true_node.value:
            result=max(self.head.true_node.classification, key=self.head.true_node.classification.get)
            #print('True',result,self.head.true_node.classification)
        else:
            result=max(self.head.false_node.classification, key=self.head.false_node.classification.get)
            #print('False',result,'Rec value',rec[self.head.true_node.column],self.head.true_node.classification)
        return result
    
    def fit(self):
        set1,set2=self.split_data_set(self.train,self.column, '1')
        clf1=self.get_uniques(set1)
        clf2=self.get_uniques(set2)
        self.head.true_node=d_node(self.column,'1',None,None,clf1)
        self.head.false_node=d_node(self.column,'1',None,None,clf2)
        
    def get_uniques(self,train):
        result={}
        if train==[]:
            return result

        for rec in train:
            if rec[0] in result.keys():
                result[rec[0]] += 1
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
    

class CoordinateDescent(object):
    def __init__(self,train,column_order=None):
        self.train = train
        self.alpha=[]
        self.trees=[]
        
        if column_order==None:
            init_alpha=1/(len(train[0])-1)
            for column in range(1,len(train[0])):
                tree=OneLevelTree(train,column)
                tree.fit()
                self.trees.append(tree)
                self.alpha.append(init_alpha)
        else:
            init_alpha=1/(len(column_order))
            for column in column_order:
                tree=OneLevelTree(train,column)
                tree.fit()
                self.trees.append(tree)
                self.alpha.append(init_alpha)

        
        #self.alpha[10]=100
    
    def fit(self,rounds=None):
        print(self.alpha)
        exp_loss=float('inf')
        if rounds is not None:
            for i in range(rounds):
                #for i,alpha in reversed(enumerate(self.alpha)):
                for i,alpha in enumerate(self.alpha):
                    self.update_alpha(i)
                new_exp_loss=self.calc_exp_loss()
                print("New exp_loss",new_exp_loss)
                if new_exp_loss<exp_loss:
                    exp_loss=new_exp_loss
            return
                
        while(1):
            for i,alpha in enumerate(self.alpha):
                self.update_alpha(i)
            new_exp_loss=self.calc_exp_loss()
            #print("New exp_loss",new_exp_loss)
            if new_exp_loss<exp_loss:
                exp_loss=new_exp_loss
            else:
                print(exp_loss,self.alpha)
                return
            
    
    def calc_exp_loss(self):
        exp_loss=0.0
        for rec in  self.train:
            exp_loss+=math.exp((-1)*rec[0]*self.predict(rec))
        
        return exp_loss
            
    def calc_coeficient(self,index_alpha,train_rec):
        '''Calculate coeficient exp(-Yi*Sum(apha*Ht(Xi))
        '''
        coef_power=0.0
        for i,tree in enumerate(self.trees):
                if i != index_alpha:
                    coef_power=coef_power+self.alpha[i]*tree.predict(train_rec)
        
        #print(index_alpha,train_rec[0],coef_power,math.exp((-1)*train_rec[0]*coef_power))
        return math.exp((-1)*train_rec[0]*coef_power) 
        
    def update_alpha(self,index_alpha):
        correct=0
        incorrect=0
        exp_upper_coef=0.0
        exp_lower_coef=0.0
        for train_rec in self.train:
            result = self.trees[index_alpha].predict(train_rec)
            if result == train_rec[0]:
                correct+=1
                exp_upper_coef += self.calc_coeficient(index_alpha, train_rec)
            else:
                incorrect+=1
                exp_lower_coef += self.calc_coeficient(index_alpha, train_rec)
        
        new_alpha = 0.5*math.log((correct*exp_upper_coef)/(incorrect*exp_lower_coef))
        #print(index_alpha,correct,incorrect,'New Alpha',new_alpha)
        self.alpha[index_alpha]=new_alpha
            
    def predict(self,test_rec):
        ''' predict single record label by using all models 
            in the tree_dictionary, by combining their results and multiplying
            each by alpha of the model
        '''
        predicted_label = 0
        for i,tree in enumerate(self.trees):
            #print(tree.predict(test_rec)*tree.alpha)
            label=tree.predict(test_rec)
            predicted_label = predicted_label + label*self.alpha[i]
        
        return predicted_label
    
    

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
    model=CoordinateDescent(train)
    
    model.fit()
    for i,tree in enumerate(model.trees):
        tree.print_tree()
        print('Alpha',model.alpha[i])
        
    correct=0
    incorrect=0
    for train_rec in train:
            result = model.predict(train_rec)
            if (train_rec[0]==1 and result>0) or (train_rec[0]==-1 and result<0):
                correct+=1
            else:
                incorrect+=1
            #print(train_rec[1],'Label',train_rec[0],'Result',result)
    print("Training Accuracy:",correct/(correct+incorrect))
    correct=0
    incorrect=0
    for train_rec in test:
            result = model.predict(train_rec)
            if (train_rec[0]==1 and result>0) or (train_rec[0]==-1 and result<0):
                correct+=1
            else:
                incorrect+=1
            #print(train_rec[1],'Label',train_rec[0],'Result',result)
    print("Testing Accuracy:",correct/(correct+incorrect))
    
    '''Play with specific order of models
    '''
    '''
    column_order=[13,11,1,7,1,8,3,22,1,16,1,20,3,8,1,11]
    model=CoordinateDescent(train,column_order)
    
    model.fit()
    for i,tree in enumerate(model.trees):
        tree.print_tree()
        print('Alpha',model.alpha[i])
        
    correct=0
    incorrect=0
    for train_rec in train:
            result = model.predict(train_rec)
            if (train_rec[0]==1 and result>0) or (train_rec[0]==-1 and result<0):
                correct+=1
            else:
                incorrect+=1
            #print(train_rec[1],'Label',train_rec[0],'Result',result)
    print("Training Accuracy:",correct/(correct+incorrect))
    correct=0
    incorrect=0
    for train_rec in test:
            result = model.predict(train_rec)
            if (train_rec[0]==1 and result>0) or (train_rec[0]==-1 and result<0):
                correct+=1
            else:
                incorrect+=1
            #print(train_rec[1],'Label',train_rec[0],'Result',result)
    print("Testing Accuracy:",correct/(correct+incorrect))
    '''
                
        
    