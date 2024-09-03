# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:34:20 2024

Exploration of MNIST Classifier from scratch

@author: ian horseman - of this version anyway

About:

    In the code below training on MNIST dataset is done using neural networks. 
    Implementation has been done with minimum use of libraries to get a better
    understanding of the concept and working on neural nets. Functions for 
    initialization, activation, forward propagation, backward propagation, 
    cost have been written separately. The training labeled dataset consists 
    of 42000 images, each of size 28x28 = 784 pixels. Labels are from 0 to 9. 
    In this we are not going to use tensorflow or any other such module.
    
    Source of orginal code bases: 
        Example 1:
        https://www.kaggle.com/code/manzoormahmood/mnist-neural-network-from-scratch
        
        Example 2:
        https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
        
    This became its own version after a few iterations with examples...I think its better!
        
--------------
Change Log
--------------
0-1 : Basic code from orginal author
0-2 : Changes to export Loss plots to Excel and clean up code
0-3 : Expanding on readibility
0-4 : Add in ability to save and load initialized layers

--------------

"""

import numpy as np
import pandas as pd
import requests, gzip, os, hashlib
#pylab inline

def init(x,y):
    layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
    return layer


#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1)    

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)


#Softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))



#forward and backward pass
def forward_backward_pass(x,y):
    targets = np.zeros((len(y),10), np.float32)
    targets[range(targets.shape[0]),y] = 1
 
    
    x_l1=x.dot(l1)
    x_sigmoid=sigmoid(x_l1)
    x_l2=x_sigmoid.dot(l2)
    out=softmax(x_l2)
   
 
    error=2*(out-targets)/out.shape[0]*d_softmax(x_l2)
    update_l2=x_sigmoid.T@error
    
    
    error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
    update_l1=x.T@error

    return out,update_l1,update_l2 


#-----------------MAIN---------------------------


#load training data and labels
train_data = pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')

#separating labels and pixels - and coverting to a numpy array. 
train_labels=np.array(train_data.loc[:,'label'])
train_data=np.array(train_data.loc[:,train_data.columns!='label'])


#------------------------------------------------
trainingset_len = train_data.shape[0]


#Validation split
rand=np.arange(trainingset_len)
np.random.shuffle(rand)
train_no=rand[:trainingset_len]

val_no=np.setdiff1d(rand,train_no)

"""
X_train,X_val=X[train_no,:,:],X[val_no,:,:]
Y_train,Y_val=Y[train_no],Y[val_no]
"""

#Initialize link weights

#set random seed for rand gereration consistancy
np.random.seed(42)

#Initialize link weights
l1_weights=init(28*28,128)
l2_weights=init(128,10)


#pull value from training set
x=0
l1_inputs = train_data[x,:]
input_label = train_labels[x]

#Example Forward Pass
l1_preactivations=l1_inputs.dot(l1_weights)
l1_activations=sigmoid(l1_preactivations)

l2_preactivations=l1_activations.dot(l2_weights)
l2_activations=softmax(l2_preactivations)


max_place_l2out = np.argmax(l2_activations)

print("Cycle result: input was ", input_label, " network answered ", max_place_l2out, " with value -> ", l2_activations[max_place_l2out])

#Mean Absolute Error (Actual - Predicted)
MAE = l2_activations[input_label] - 1
print("MAE is: ", MAE)


#Example Backward Pass
