'''
Created on Aug 31, 2016

@author: uri
Implement Stochastic gradient descent where exactly one component of the sum is chosen to approximate
the gradient at each iteration. Instead of picking a random component at each iteration, iterate through the data set starting with the first element, 
then the second, and so on until the nth element, at which back at the beginning again.
Use the step size  t = 1.
Descent procedure should start from the initial point
w0 = [0,0,0,0]
b0 = 0
'''
import numpy as np
from numpy import sign

def perceptron_opt_batch(train_df,step,np_w,b):
    i=0
    while True:
        i+=1
        w_total_correction=np.array([0,0,0,0])
        b_correction=0
        for rec in train_df:
            rec_x = rec[0:4] #np.insert(rec[0:4],0,1) #insert 1 in the first position for w0 
            y=rec[4]
            if (y !=sign(np.dot(np_w,rec_x) + b)):
                w_correction=rec_x*y
                w_total_correction=w_total_correction+w_correction
                b_correction+=y
                #print('Record, and correction',rec,"...",w_correction,"....",b_correction)
        
        if np.array_equal(w_total_correction,np.array([0,0,0,0])):
            print("Algorithm converged;number of loops and , final weights",i,'...',np_w,b)
            break
        else:
            if i>100:
                print("Algorithm did not converge",np_w,b)
                break
        
        np_w=np_w+step*w_total_correction
        b=b+step*b_correction
        print("After iteration completion, new weights..",np_w,'....',b)
        
    return i,np_w,b

if __name__ == '__main__':
    
    train_df = np.genfromtxt('data/perceptron.data',delimiter=',')
    np_w=np.array([0,0,0,0])
    b=0
    step=1
    opt_step=0
    iter_m=100
    iter,np_w,b=perceptron_opt_batch(train_df,1,np_w,b)
    '''
    for i in range(1,100):
        iter,np_w,b=perceptron_opt_batch(train_df,i,np_w,b)
        if iter < iter_m:
            opt_step=i
            iter_m=iter
        np_w=np.array([0,0,0,0])
        b=0
    print("Min i is ..",opt_step,iter_m)
    '''