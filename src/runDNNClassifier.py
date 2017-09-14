# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:59:50 2017

@author: rgast
"""
"""
This script writes results of a MLP runthrough (Performances and evidences) in a pickle file MLP_Results.pickle.
@return: writes results of a MLP runthrough (Performances and evidences) in a pickle file MLP_Results.pickle
@rtype: pickle file
"""
""" Libraries """

from matplotlib.pyplot import *
import os
import argparse
import sys
import pickle

"""
this weird section of code allows modules in the parent directory to be imported here
it's the only way to do it in a way that allows you to run the file from other directories
and still have it work properly
"""
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from DeepNN import DeepNN
import preprocessing

import numpy as np
import random

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=np.ComplexWarning)

# set random seeds for both numpy and random
SEED = 100
np.random.seed(SEED)
random.seed(SEED)


""" Functions """

def runDNN(path, syllN, trainN, cvalRuns, sampRate, interpolType, mfccN, invCoeffOrder, winsize, melFramesN,
        smoothL, polyOrder, incDer, snr = 0.0, syllNames = None, layerSizes = [60,10], activationFcts = 'tanh', 
        dropouts = [], normalizations = [], optimizer = 'Adam', learningRate = 0.0005, batchSize = 10, 
        nEpochs = 10, loss = 'CrossEntropyExclusiveSparse', validate_per_step = 100, samplingSDs = 0.05):
    """
    Function that runs syllable classification in a supervised manner using positive, negative and combined
    conceptors.

    @type path: string
    @param path: directory to the folder that includes syllable folders with wave data
    @type syllN:
    @param syllN:
    @type trainN: integer
    @param trainN: number of training samples to use for each syllable
    @type cvalRuns: integer
    @param cvalRuns: Number of cross validation runs with different training/test data splits
    @type sampRate: integer
    @param sampRate: Sampling Rate that raw data will be downsampled to
    @type interpolType: string
    @param interpolType: type of interpolation to be used for downsampling
    @type mfccN: integer
    @param mfccN: Number of mel frequency cepstral coefficients to extract for each mel frame
    @type invCoeffOrder: boolean
    @param invCoeffOrder: if true: Extract last n mfcc instead of first n
    @type winsize: integer
    @param winsize: Size of the time-window to be used for mfcc extraction in ms
    @type melFramesN: integer
    @param melFramesN: Desired number of time bins for mfcc data
    @type smoothL: integer
    @param smoothL: Desired length of the smoothed mfcc data
    @type polyOrder: integer
    @param polyOrder: Order of the polynomial used for mfcc data smoothing
    @type incDer: List of 2 booleans
    @param incDer: indicating whether to include 1./2. derivative of mfcc data or not
    @type snr: float
    @param snr: signal to noise ratio in the syllable data (if 0, no noise is added )
    @type syllNames: List of strings
    @param syllNames: List of names of specific syllables to be used
    @type layerSizes: List of integers
    @param layerSizes: 
    @type activationFcts: List of strings
    @param activationFcts: Activation function for units in each layer
    @type dropouts: List of floats
    @param dropouts: weight dropout probability for each layer
    @type normalizations: List of ?
    @param normalizations: indicates which normalization to add to layer activations
    @type optimizer: string
    @param optimizer: optimizer to use for gradient descent
    @type learningRate: float
    @param learningRate: the learning rate of the gradient descent optimizer
    @type batchSize: integer
    @param batchSize: batch size for each weight update
    @type nEpochs: integer
    @param nEpochs: how many times to go through all training data to train MLP
    @type loss:
    @param loss: 
    @type validate_per_step: integer
    @param validate_per_step: after how many training steps to evaluate performance on validation set
    @type samplingSDs: List of floats
    @param samplingSDs: standard deviation of gaussian from which initial weights are sampled
    @rtype:
    @return: 
    
    """
    
    path = os.path.abspath(path)

    """ assign parameters """
    
    prepParams = {
        'syll_names': syllNames,
        'sample_rate': sampRate,
        'ds_type': interpolType,
        'mel_channels': mfccN,
        'inv_coefforder': invCoeffOrder,
        'winsize': winsize,
        'frames': melFramesN,
        'smooth_length': smoothL,
        'inc_der': incDer,
        'poly_order': polyOrder,
        'snr': snr
    }

    performances = []
    evidences = []

    for i in range(cvalRuns):

        n_test = np.ones(syllN, dtype = int)*20
        
        Samples = []
        if cvalRuns > 1:
            for j in range(syllN):
    
                indices = np.arange(0, trainN + n_test[j], 1)
                ind_tmp = indices.copy().tolist()
                random.shuffle(ind_tmp)
                ind_tmp = np.array(ind_tmp)
                
                Samples.append(ind_tmp)

        """ Get and preprocess data """
        
        data = preprocessing.preprocess(path, syllN, trainN, n_test, samples = Samples, **prepParams)
        trainData = data['train_data']
        testData = data['test_data']
        
        inpDim = mfccN*(1+sum(incDer))*smoothL
        testL = int(n_test[0]/2)
        data_train = np.zeros((len(trainData)*trainN, inpDim))
        labels_train = np.zeros(len(trainData)*trainN)
        data_test = np.zeros((len(testData)*testL, inpDim))
        labels_test = np.zeros(len(testData)*testL)
        data_validate = np.zeros_like(data_test)
        labels_validate = np.zeros_like(labels_test)
        
        for t,syll in enumerate(trainData):
            
            for s,sample in enumerate(syll):
            
                data_train[t*trainN+s,:] = sample.flatten()
                labels_train[t*trainN+s] = t
        
        for t,syll in enumerate(testData):
            
            for s,sample in enumerate(syll):
                
                if s < n_test[0]/2:
                    
                    data_test[t*testL+s,:] = sample.flatten()
                    labels_test[t*testL+s] = t
                
                else:
                    
                    data_validate[t*testL+s-testL,:] = sample.flatten()
                    labels_validate[t*testL+s-testL] = t
        
        """ create DNN """
        
        syllClassifier = DeepNN(inpDim, 1)
        
        if not any(dropouts):
            
            dropouts = np.zeros(len(layerSizes))
            
        if not any(normalizations):
            
            normalizations = np.zeros(len(layerSizes))
        
        if not type(samplingSDs) == np.ndarray:
            
            samplingSDs = np.zeros(len(layerSizes)) + samplingSDs
            
        for n,l in enumerate(layerSizes):
            
            if n == 0:
                
                if type(activationFcts) is not list:
                    
                    syllClassifier.addLayer(l, activationFcts, include_bias = True, normalization = normalizations[n], dropout = dropouts[n], sd = samplingSDs[n])  
                    
                else:
                    
                    syllClassifier.addLayer(l, activationFcts[n], include_bias = True, normalization = normalizations[n], dropout = dropouts[n], sd = samplingSDs[n])
                    
            else:
            
                if type(activationFcts) is not list:
                    
                    syllClassifier.addLayer(l, activationFcts, include_bias = False, normalization = normalizations[n], dropout = dropouts[n], sd = samplingSDs[n])  
                    
                else:
                    
                    syllClassifier.addLayer(l, activationFcts[n], include_bias = False, normalization = normalizations[n], dropout = dropouts[n], sd = samplingSDs[n])
        
        """ train DNN and classify test data """
        
        data = [data_train,data_validate]
        labels = [labels_train, labels_validate]
        syllClassifier.train(data, labels, loss_type = loss, optimizer_type = optimizer, learning_rate = learningRate, n_epochs = nEpochs, 
                             batch_size = batchSize, validate_per_step = validate_per_step, verbose = 20)
        syllClassifier.test(data_test, labels_test, normalize = True)
        results = syllClassifier.test_predictions
        performance = np.mean(np.argmax(results, axis = 1) == labels_test)
        evidences.append(results)
        performances.append(performance)

    cval_results = {'Evidences': evidences, 'Labels': labels_test, 'Performances': performances}
    
    return cval_results
    
    
""" MLP parameters """

path = '/home/rgast/Documents/GitRepo/BirdsongRecog/data/birddb/syll' # directory to the folder that includes syllable folders with wave data
trainN = 50 # number of training samples to use for each syllable
cvalRuns = 10 # Number of cross validation runs with different training/test data splits
sampRate = 20000 # Sampling Rate that raw data will be downsampled to
interpolType = 'mean' # type of interpolation to be used for downsampling
mfccN = 20 # Number of mel frequency cepstral coefficients to extract for each mel frame
invCoeffOrder = True # Boolean, if true: Extract last n mfcc instead of first n
winsize = 20 # Size of the time-window to be used for mfcc extraction in ms
melFramesN = 64 # Desired number of time bins for mfcc data
smoothL = 4 # Desired length of the smoothed mfcc data
polyOrder = 3 # Order of the polynomial used for mfcc data smoothing
incDer = [True,True] # List of 2 booleans indicating whether to include 1./2. derivative of mfcc data or not
targetDir = None # Subdirectory in which results are to be stored
syllNames = None # List of names of specific syllables to be used
snr = 0.0 # signal to noise ratio in the syllable data (if 0, no noise is added )
layerSize = mfccN * smoothL * (1 + np.sum(np.array(incDer)))
learningRate = 0.001 # the learning rate of the gradient descent optimizer
batchSize = 10 # batch size for each weight update
nEpochs = 3 # how many times to go through all training data to train MLP
optimizer = 'Adam' # optimizer to use for gradient descent
activationFcts = ['tanh','none'] # Activation function for units in each layer
dropouts = [0.,0.] # weight dropout probability for each layer
validatePerStep = 10 # after how many training steps to evaluate performance on validation set
loss = 'CrossEntropyExclusiveSparse' # loss to use for gradient calculation
samplingSDs = [0.1, 0.1] # standard deviation of gaussian from which initial weights are sampled
normalizations = [None,None] # indicates which normalization to add to layer activations

""" evaluate MLP performance for different numbers of syllables """

n_syllables = np.arange(50,60,5)

performances = []
evidences = []

for n in n_syllables:
    
    layerSizes = [layerSize,n]

    cval_results = runDNN(path=path, syllN=n, trainN=trainN, cvalRuns=cvalRuns,
        sampRate=sampRate, interpolType=interpolType, mfccN=mfccN,
        invCoeffOrder=invCoeffOrder, winsize=winsize, melFramesN=melFramesN,
        smoothL=smoothL, polyOrder=polyOrder, incDer=incDer, snr=snr, 
        syllNames = syllNames, layerSizes = layerSizes, learningRate = learningRate, 
        batchSize = batchSize, nEpochs = nEpochs, optimizer = optimizer,
        activationFcts = activationFcts, dropouts = dropouts, validate_per_step = validatePerStep,
        loss = loss, samplingSDs = samplingSDs, normalizations = normalizations)
        
    performances.append(cval_results['Performances'])
    evidences.append(cval_results['Evidences'])

""" save results """

with open('MLP_Results.pickle', 'wb') as f:
    
    pickle.dump([performances, evidences], f)
    