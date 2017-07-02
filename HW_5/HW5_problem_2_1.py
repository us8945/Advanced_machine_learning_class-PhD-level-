'''
Created on Nov 29, 2016

@author: Uri Smashnov
Expectation Maximization 
For this problem, you will used the data provided in bn.data. This dataset was generated from
the following Bayesian network.
                    > B  \
				A /       \ > D  
			      \       /
				     > C /
					 
The columns in the dataset correspond, in order, to the four binary random variables xA; xB; xC; xD.
The goal of this problem is to learn the parameters of this Bayesian network from the provided
data. However, the observations in the training set can contain missing data. 
In this problem, we will assume that the data is missing completely at random and that the probability that attribute i is missing is independent of whether or not attribute j is missing. 
The probability that attribute i is missing is equal to an unknown probability, bi.

Use the EM algorithm to learn both the parameters of the BN and the probabilities b1; :::; b4.

'''
import csv
from builtins import int

def calculate_p(train,col,dep_col=None):
    count=0
    count_dep_zero=0
    count_rec_dep_zero=0
    count_dep_one=0
    count_rec_dep_one=0
    
    count_dep_zero_zero=0
    count_dep_zero_one=0
    count_dep_one_zero=0
    count_dep_one_one=0
    count_rec_dep_zero_zero=0
    count_rec_dep_zero_one=0
    count_rec_dep_one_zero=0
    count_rec_dep_one_one=0
    
    prob={}
    if dep_col==None:
        for rec in train:
            #print(rec[col])
            count+=rec[col]
        return count/len(train)
    if type(dep_col)==int: ##Only onde dependency
        for rec in train:
            if rec[dep_col]==0:
                count_dep_zero+=1
                if rec[col]==1:
                    count_rec_dep_zero+=1
            if rec[dep_col]==1:
                    count_dep_one+=1
                    if rec[col]==1:
                        count_rec_dep_one+=1
        #print(count_rec_dep_zero,count_dep_zero,count_rec_dep_one,count_dep_one)
        return((count_rec_dep_zero/count_dep_zero,count_rec_dep_one/count_dep_one))
     
    if type(dep_col)==tuple:
        for rec in train:
            if (rec[dep_col[0]],rec[dep_col[1]])==(0,0):
                count_dep_zero_zero+=1
                if rec[col]==1:
                    count_rec_dep_zero_zero+=1
            if (rec[dep_col[0]],rec[dep_col[1]])==(0,1):
                count_dep_zero_one+=1
                if rec[col]==1:
                    count_rec_dep_zero_one+=1
            if (rec[dep_col[0]],rec[dep_col[1]])==(1,1):
                count_dep_one_one+=1
                if rec[col]==1:
                    count_rec_dep_one_one+=1
            if (rec[dep_col[0]],rec[dep_col[1]])==(1,0):
                count_dep_one_zero+=1
                if rec[col]==1:
                    count_rec_dep_one_zero+=1
        return(count_rec_dep_zero_zero/count_dep_zero_zero, count_rec_dep_zero_one/count_dep_zero_one, count_rec_dep_one_zero/count_dep_one_zero, count_rec_dep_one_one/count_dep_one_one )
                   

def estimate_completions(train_mis,P_a,P_b,P_c,P_d):
    '''Estimate left to right, this will insure records with multiple missing values are estimated
        int(str(a)+str(b),2) - concatenate two strings and convert to integer using binary base. Example: '01' =1, '11'=3
    '''
    new_train_compl=[]
    
    for rec in train_mis:
        if rec.count('?') > 1:
            continue
        elif rec[0]=='?':
            rec[0] = P_a[int(str(rec[1])+str(rec[2]),2)]
        elif rec[1]=='?':
            #print(P_b,rec[0])
            rec[1] = P_b[int(rec[0])]
        elif rec[2]=='?':
            rec[2] = P_c[int(rec[0])]
        elif rec[3] =='?':
            #print(P_d,int(str(rec[1])+str(rec[2]),2))
            rec[3] = P_d[int(str(rec[1])+str(rec[2]),2)]
        
        new_train_compl.append([float(rec[0]),float(rec[1]),float(rec[2]),float(rec[3])])
    #print(new_train_compl)
    return new_train_compl

f=open('data/bn.data')
reader = csv.reader(f, delimiter=',') 
train=[]
for rec in reader:
    #rec=rec[0:2] #for testing purposes
    train.append(rec)
f.close()

print(train)

train_mis=[]
train_compl=[]
for rec in train:
    for i,var in enumerate(rec):
        if var=='?':
            train_mis.append(rec)
            break
        else:
            if i==3:
                train_compl.append([int(rec[0]),int(rec[1]),int(rec[2]),int(rec[3])])

print(len(train_mis), train_mis)
print(len(train_compl),train_compl)

'''Step Initialize P(A), P(B/A), P(C/A), P(D/B,C)
'''
P_a=calculate_p(train_compl,0,(1,2))
print('P(A/B,C):',P_a)
P_b=calculate_p(train_compl,1,0)
print(('P(B/A)'),P_b)
P_c=calculate_p(train_compl,2,0)
print('P(C/A)',P_c)
P_d = calculate_p(train_compl,3,(1,2))
print('P(D/B,C',P_d)

'''Step E - estimate completions and M in Loop 20
'''
for i in range (100):
    print("***********Iteration:",i+1,'  *****************')
    train_mis_new = estimate_completions(train_mis,P_a,P_b,P_c,P_d)
    
    train_new = train_compl+train_mis_new
    P_a=calculate_p(train_new,0,(1,2))
    print('P(A):',P_a)
    P_b=calculate_p(train_new,1,0)
    print(('P(B)'),P_b)
    P_c=calculate_p(train_new,2,0)
    print('P_c',P_c)
    P_d = calculate_p(train_new,3,(1,2))
    print('P_d',P_d)


''' Final probability for A
'''
count_a=0
count_all=0
for rec in train_compl+train_mis_new:
    if rec[0]==1:
        count_a+=1
    if rec[0]!='?':
        count_all+=1

print("Final probability for A=1 is ",count_a/count_all)

''' Estimate probability of value missing assuming "missing completely at random"
'''
missing=[0,0,0,0]

for rec in train:
    for i,var in enumerate(rec):
        if var=='?':
            missing[i]+=1


'''MLE calculations
'''
print("***************MLE Calculations******************")
print("Probability A is missing:", round(missing[0]/len(train_mis),2))
print("Probability B is missing:", round(missing[1]/len(train_mis),2))
print("Probability C is missing:", round(missing[2]/len(train_mis),2))
print("Probability D is missing:", round(missing[3]/len(train_mis),2))

train_inp=[]
for rec in train:
    if rec[0]!='?' and rec[1]!='?' and rec[2]!='?':
        train_inp.append([int(rec[0]),int(rec[1]),int(rec[2]),rec[3]])
    
P_a=calculate_p(train_inp,0,(1,2))
print('P(A/B,C):',P_a)

train_inp=[]
for rec in train:
    if rec[0]!='?' and rec[1]!='?':
        train_inp.append([int(rec[0]),int(rec[1]),rec[2],rec[3]])
    
P_b=calculate_p(train_inp,1,0)
print(('P(B/A)'),P_b)

train_inp=[]
for rec in train:
    if rec[0]!='?' and rec[2]!='?':
        train_inp.append([int(rec[0]),rec[1],int(rec[2]),rec[3]])
    
P_c=calculate_p(train_inp,2,0)
print('P(C/A)',P_c)

train_inp=[]
for rec in train:
    if rec[1]!='?' and rec[2]!='?' and rec[3]!='?':
        train_inp.append([rec[0],int(rec[1]),int(rec[2]),int(rec[3])])
        
P_d = calculate_p(train_inp,3,(1,2))
print('P(D/B,C)',P_d)



'''Step Initialize P(A), P(B/A), P(C/A), P(D/B,C)
'''
print("************Initialize probabilities differently*************")

P_a=(0,0,0,0)
print('P(A/B,C):',P_a)
P_b=(0,0)
print(('P(B/A)'),P_b)
P_c=(0,0)
print('P(C/A)',P_c)
P_d = (0,0,0,0)
print('P(D/B,C',P_d)

'''Step E - estimate completions and M in Loop 100
'''
for i in range (100):
    
    train_mis_new = estimate_completions(train_mis,P_a,P_b,P_c,P_d)
    #print(train_mis_new)
    train_new = train_compl+train_mis_new
    P_a=calculate_p(train_new,0,(1,2))
    P_b=calculate_p(train_new,1,0)
    P_c=calculate_p(train_new,2,0)
    P_d = calculate_p(train_new,3,(1,2))
    
print('***********Iteration: 100 *****************')
print('P(A):',P_a)
print(('P(B)'),P_b)
print('P_c',P_c)
print('P_d',P_d)