'''
Created on Aug 31, 2016

@author: uri
Implement the perceptron algorithm using Standard gradient descent with the step size  t = 1 for each iteration.
Descent procedure should start from the initial point
w0 = [0,0,0,0]
b0 = 0
'''
import numpy as np
from numpy import sign

if __name__ == '__main__':
    train_df = np.genfromtxt('data/perceptron.data',delimiter=',')
    train_y = train_df[4]
    delta=0
    np_w=np.array([0,0,0,0])
    b=0
    step=1
    i=0
    while True:
        i+=1
        correction=False
        #w_total_correction=np.array([0,0,0,0])
        #b_correction=0
        delta=0
        for rec in train_df:
            rec_x = rec[0:4] #np.insert(rec[0:4],0,1) #insert 1 in the first position for w0 
            y=rec[4]
            if (y !=sign(np.dot(np_w,rec_x) + b)):
                correction=True
                #w_correction=rec_x*y
                #w_total_correction=w_total_correction+w_correction
                #b_correction=y
                #np_w=np_w+step*w_total_correction
                np_w=np_w+step*rec_x*y
                b=b+step*y
                #print('Record, and correction',rec,"...",w_correction,"....",b_correction)
        
        if correction==False:
            print("Algorithm converged on iteration",i,", final weights",np_w,b)
            break
        else:
            if i>100:
                print("Algorithm did not converge",np_w,b)
                break
        
        print("Iteration",i,"completion, weights",np_w,' b coefficient',b)
                
        
    