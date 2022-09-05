#!/usr/bin/env python
# _*_coding:utf-8_*_
import matplotlib.pylab as plt
import numpy as np
import keras
import os
import tensorflow as tf
from keras.datasets.fashion_mnist import load_data
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten,Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
import pandas as pd
import simplejson
from sklearn.metrics import auc,roc_auc_score,roc_curve
from sklearn.metrics import matthews_corrcoef

def get_NCP(R):
    if (R == 'A'):
        return np.array([1,1,1])
    if (R == 'C'):
        return np.array([0,1,0])
    if (R == 'G'):
        return np.array([1,0,0])
    if (R == 'T'):
        return np.array([0,0,1])  

def readFile_x(sequences): #Please give a sequence in the fasta format. 
    NCP_ND = []
    Sequence_list = []
    for i in range(len(sequences)):
        if (i % 2 == 1):
            sequence_i = sequences[i]
            sequence_i = sequence_i.strip('\n')
            sequence_i = sequence_i.strip('"')
            sequence_i = sequence_i.lstrip()
            Sequence_list.append(sequence_i)
            for j in range(len(sequence_i)):
                ND_j = sequence_i.count(sequence_i[j], 0, j+1)/(j+1)
                NCP_j = get_NCP(sequence_i[j])
                NCP_ND_j = [NCP_j[0], NCP_j[1], NCP_j[2], ND_j]
                if j == 0:
                    NCP_ND_i = NCP_ND_j
                if j != 0:
                    NCP_ND_i = np.vstack((NCP_ND_i, NCP_ND_j))
         
            NCP_ND.append(NCP_ND_i)   
        
    NCP_ND = np.array(NCP_ND)
    return(NCP_ND)

def readFile_y(file):
    label_array = []
    for iS in range(len(file)):
        label_i = file[iS]
        label_i = label_i.strip('\n') 
        label_array.append(float(label_i))
   
    return np.transpose(label_array)    

def DPred_model(window_half, element_size, lr, num_classes):  
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=1, input_shape=(window_half*2+1, element_size)))
    model.add(SeqSelfAttention(attention_activation='softmax', attention_width = 2))
    model.add(tf.keras.layers.Reshape((window_half*2+1, element_size, 1),))
    model.add(Conv2D(100, (2, 2), strides=(2, 2),padding = 'same', activation='relu', input_shape=(window_half*2+1, element_size, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))  
    model.compile(optimizer=keras.optimizers.Adam(lr = lr), loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

def Aucs(X_test, model, y_test):
    prediction = model(X_test) #X_test here is the output of readFile_x
    y_hat_test = np.argmax(prediction, axis=1)
    AUC = roc_auc_score(y_test, y_hat_test) #y_test here is the output of readFile_y
    return AUC

def Mccs(X_test, model, y_test):
    prediction = model(X_test)
    y_hat_test = np.argmax(prediction, axis=1)
    MCC = matthews_corrcoef(y_test, y_hat_test)
    return MCC

def Roc_curves(X_test, model, y_test):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prediction = model(X_test)
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(
    fpr[1],
    tpr[1],
    color="aqua",
    linewidth = 2)
    
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    
def summarized_results(prediction, Sequence_list):
    prob = np.array(prediction[:, 1])
    y_prediction = np.argmax(prediction, axis=1)
    Prediction_Result = []
    Sequence_index = []
    for i in range(len(y_prediction)):
        Sequence_index.append('sequence_' + str(i+1))
        if (y_prediction[i] == 1):
            Prediction_Result.append('DPred positive (modified) site')
        if (y_prediction[i] == 0):
            Prediction_Result.append('DPred negative (non-modified) site')

    dictionary = []
    for i in range(len(y_prediction)):
        dictionary_i = {'Job ID': 'jobID', 
                        'Index': Sequence_index[i],
                        'Sequence': Sequence_list[i],
                        'Probability': format(prob[i], '.4f'), 
                        'Prediction Result': Prediction_Result[i]
                        }
        dictionary.append(dictionary_i)
    return(dictionary)

def run_an_example(PATH):
    model = load_model(PATH + '/model.h5',  custom_objects={'SeqSelfAttention': SeqSelfAttention})
    file_data = open(PATH + '/example.fasta') 
    sequences = file_data.readlines()
    X_test = readFile_x(sequences)
    prediction = model(X_test)
    return(prediction)
 






