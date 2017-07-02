'''
Created on Dec 3, 2016

Home work 6
Problem 3
@author: Uri Smashnov
Neural Networks
Purpose: build and train Neural Net to classify odd/even scenario. If number of "ones" is even than classify as "1", otherwise "0".
Flow: 
- generate training set with all possible combinations of "one's" and "zeros"
- Build NN network with:
     - One input layer (pass through) with number of neurons as number of input digits
     - Two hidden Layers of Sigmoid Perceptrons. Each layer has number of neurons as number of input digits
     - One output layer consisting of two Softmax neurons for classification
     - The output is classified as "zero"
'''
from pybrain import * 
import itertools
import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer 


def generate_training_set(dimensions, mult=1):
    ''' Returns training set and Pybrain ClassificationDataSet class'''
    P_data_set = ClassificationDataSet(dimensions, 1 , nb_classes=2)
    #P_data_set = SupervisedDataSet(dimensions, 1)
    combinations = ["".join(seq) for seq in itertools.product("01", repeat=dimensions)]
    train_x=[]
    train_y=[]
    for i in range(mult):
        for rec in combinations:
            r=list(map(int,list(rec)))
            if (sum(r))==0:
                label=0
            else:
                label= (sum(r)+1)%2
            #print(label,r)
            train_x.append(r)
            train_y.append(label)
     
    for k in range(len(train_y)): 
        P_data_set.addSample(np.ravel(train_x[k]),train_y[k])
    
    '''classification label is list of two. If it is [1 0] than output is classified as "0", 
       if it is [0 1] it is classified as "1". This is done with _convertToOneOfMany() '''
    P_data_set._convertToOneOfMany()
    
    return P_data_set, train_x, train_y
 


    
'''Set training set dimensions and how many times data should be multiplied'''
dimensions=10
mult=1

''' Define Neural Network '''    
nn2 = FeedForwardNetwork(name='NNet_2_Layers')
nn2.addInputModule(LinearLayer(dimensions, name='in'))
nn2.addModule(SigmoidLayer(int(dimensions), name='hidden1'))
nn2.addModule(SigmoidLayer(int(dimensions), name='hidden2'))
nn2.addOutputModule(SoftmaxLayer(2, name='out'))
nn2.addConnection(FullConnection(nn2['in'], nn2['hidden1'], name='c1'))
nn2.addConnection(FullConnection(nn2['hidden1'], nn2['hidden2'], name='c2'))
nn2.addConnection(FullConnection(nn2['hidden2'], nn2['out'], name='c3'))
nn2.sortModules()


""" Print the NN structure: """
print(nn2)

''' Generate dataset and define "trainer" class'''

ds,train_x,train_y=generate_training_set(dimensions,mult)
ds1,train_x,train_y=generate_training_set(dimensions,mult)
trainer = BackpropTrainer( nn2, ds )
trainer.train()
trainer.trainUntilConvergence(trainingData=ds, validationData=ds1)
predict_y=trainer.testOnClassData(dataset=ds,verbose=True)
correct=0
for i,p in enumerate(predict_y):
    if p==train_y[i]:
        correct+=1
    else:
        print("Incorrectly classified sample",train_x[i],"Predicted:",predict_y, "Correct:",train_y[i])

print("Training set with dimensions:",dimensions," multiply by:",mult," Accuracy is:",correct/len(train_y))

''' Demonstrate impact of dataset size using dimension=4'''
dimensions=6
''' Define Neural Network '''    
for mult in range(1,1):
    ds,train_x,train_y=generate_training_set(dimensions,mult)
    trainer = BackpropTrainer( nn2, ds )
    trainer.trainUntilConvergence()
    predict_y=trainer.testOnClassData(dataset=ds,verbose=True)
    correct=0
    for i,p in enumerate(predict_y):
        if p==train_y[i]:
            correct+=1
    
    print("Training set with dimensions:",dimensions," Dataset size:", len(train_y)," Accuracy is:",correct/len(train_y))

'''
Training set with dimensions: 6  Dataset size: 64  Accuracy is: 0.515625
Training set with dimensions: 6  Dataset size: 128  Accuracy is: 0.984375
Training set with dimensions: 6  Dataset size: 192  Accuracy is: 0.484375
Training set with dimensions: 6  Dataset size: 256  Accuracy is: 0.46875
Training set with dimensions: 6  Dataset size: 320  Accuracy is: 0.96875
Training set with dimensions: 6  Dataset size: 384  Accuracy is: 0.53125
'''
