# Desciption: A Denoising autoencoder built using Theano for dimensionality reduction
# or pretraining s deep neural network 

import numpy as np
import sys
import os
import theano
import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
import time
from preprocess import *
import matplotlib.pyplot as plt


class AutoEncoder:
    
    def __init__(self,input,n_vis,n_hid):
        '''
        A denoising autoencoder to extract hidden features from high dimensional input
        '''
        # Random number generators
        self.numpy_rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
        
        # Symbolic input
        self.x = input
        
        # Number of visible and hidden units
        self.n_vis = n_vis
        self.n_hid = n_hid

        # Initialize weights
        W = np.asarray(self.numpy_rng.uniform(
                  low=-4* np.sqrt(6./(n_hid+self.n_vis)),
                  high=4* np.sqrt(6./(n_hid+self.n_vis)),
                  size=(self.n_vis, n_hid)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W,name='W')
        self.W_prime = self.W.T
        
        # Initialize visible biases
        self.vbias = theano.shared(value=np.zeros(self.n_vis,
                                        dtype=theano.config.floatX), name='vbias')
        
        # Initialize hidden biases
        self.hbias = theano.shared(value=np.zeros(n_hid,
                                             dtype=theano.config.floatX), name='hbias')
        
        # Learnable Paramters
        self.params = [self.W,self.hbias,self.vbias]
        
    def corrupt(self,x,corruption_level):
        '''
        Adds some random noise to the input
        '''
        return self.theano_rng.binomial(size=x.shape,n=1,p=1-corruption_level)*x

    def encode(self,input):
        '''
        Returns extracted code from input
        '''
        return T.nnet.sigmoid(T.dot(input,self.W)+self.hbias)

    def decode(self,hidden):
        '''
        Reconstructs the input
        '''
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime)+self.vbias)

    def updateModel(self,lr,dA=True):
        '''
        Returns Cross-Entropy error and parameter updates
        '''

        y = self.encode(self.x)
        z = self.decode(y)
        
        # Cross Entropy Error
        CE = T.mean(-T.sum(self.x*T.log(z),axis=1))
        
        # Gradients
        grad_params = T.grad(CE,self.params)
        
        # Parameter updates
        updates = []
        for param, gparam in zip(self.params, grad_params):
            updates.append((param,param-lr*gparam))
        return CE,updates,y

def trainAE(dataset,lr=0.01,batch_size=10,n_hid=850,num_epochs=20,noise=0.2):
    '''
    Trains the AutoEncoder and extract hidden representation from the data
    '''
    sys.stdout.write('Training a Denoising Autoencoder...\n')
     
    # Create the dataset 
    training_dataset = theano.shared(dataset[:int(0.8*dataset.shape[0]),:])
    valid_dataset = theano.shared(dataset[int(0.8*dataset.shape[0]):,:])
    
    # Create a tensor-type scalar to index the dataset
    index = T.lscalar()

    # Calculate number of batches based on batch_size
    num_batches = training_dataset.get_value().shape[0]//batch_size
    num_batches_valid = valid_dataset.get_value().shape[0]//batch_size

    # Define a theano tensor matrix
    x = T.matrix('x')
    
    # Creates an autoencoder class 
    model = AutoEncoder(x,dataset.shape[1],n_hid)
    
    # Define a reference to get the error and hidden representations
    error,updates,codes  = model.updateModel(lr)
    
    # Define a function to train the data using stochastic gradient descent
    train = function([index],error,updates=updates,givens={x:model.corrupt(training_dataset[index*batch_size:batch_size*(index+1)],noise)})
    
    validate = function([index],error,givens={x:valid_dataset[index*batch_size:batch_size*(index+1)]})

    c = model.encode(x)
    getCodes = function([x],c)
    
    # Get mean error per epoch
    CE = np.zeros((num_epochs,1))
    CE_valid = np.zeros((num_epochs,1))
    for epoch in xrange(num_epochs):
        error = []
        valid_err=[]
        for batch in xrange(num_batches):
            sys.stdout.write('Epoch %d \tTrainingBatch %d\n'%(epoch,batch))
            error.append(train(batch))
        for v_batch in xrange(num_batches_valid):
            sys.stdout.write('Epoch %d \tValidationBatch %d\n'%(epoch,v_batch))
            valid_err.append(validate(v_batch))
        
        CE[epoch] = np.mean(error)
        CE_valid[epoch] = np.mean(valid_err)
        sys.stdout.write('Epoch %d \tTraining_Error: %.5f\tValidation_Error: %.5f\n'%(epoch,CE[epoch],CE_valid[epoch]))
    
    # Plot the errors
    plt.figure(1)
    plt.plot(range(num_epochs),CE,'r',label='Training')
    plt.plot(range(num_epochs),CE_valid,'b',label='Validation')
    plt.xlabel('Epochs');plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    # Get hidden codes to be used for supervised learning
    hidden_units = getCodes(dataset)
    sys.stdout.write('Successfuly extracted the hidden codes...')
    return hidden_units
    


    
