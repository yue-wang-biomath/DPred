#!/usr/bin/env python
# _*_coding:utf-8_*_

# Codes for Transformer were implemented with Pytorch 1.2
# Transformer
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

torch.set_default_tensor_type(torch.DoubleTensor)

# load dataset
def load_sequence(Xdataset, ydataset, N_seq = 41):
    Xydataset = []
    for iS in range(Xdataset.shape[0]):
        Xdataset_i = Xdataset[iS]
        ydataset_i = ydataset[iS]
        Xydataset_i = (torch.tensor(np.transpose(Xdataset_i.reshape(N_seq, 4)).reshape(1, N_seq, 4)), ydataset_i)
        Xydataset.append(Xydataset_i)
        
    return Xydataset

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def actfunc(activation='relu'):
    if activation == 'relu':
        act = nn.ReLU
    return act()

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, len=None, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.len = len
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position *= 2
        div_term = torch.exp(position / d_model * math.log(10000))
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe.requires_grad = False
        pe = pe / 10
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2)]
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, inputsize, headnum=8, modelsize=None):
        super(SelfAttention, self).__init__()
        if modelsize is None:
            modelsize = inputsize // headnum
        self.Wq = clones(nn.Linear(inputsize, modelsize, bias=False), headnum)
        self.Wk = clones(nn.Linear(inputsize, modelsize), headnum)
        self.Wv = clones(nn.Linear(inputsize, modelsize), headnum)
        self.size = 1 / (modelsize ** 0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, m=None):
        z = []
        if m is None:
            m = x
        for i in range(len(self.Wq)):
            q = self.Wq[i](x)
            k = self.Wk[i](m).transpose(-1, -2)
            weight = torch.mul(torch.matmul(q, k), self.size)
            v = torch.matmul(self.softmax(weight), self.Wv[i](m))
            z.append(v)
        z = torch.cat(z, -1)
        return z
    
class Conv1dtranspose(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, dilation=1, padding=0, pooling=False,
                 in_transpose=False, out_transpose=False, groups=1, dropout=0.1,acti='relu'):
        super(Conv1dtranspose, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, padding=padding, groups=groups,
                              kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.in_transpose = in_transpose
        self.out_transpose = out_transpose
        self.dropout = nn.Dropout(dropout)
        self.out=actfunc(acti)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x, in_transpose=False):
        if in_transpose:
            x = torch.transpose(x, -1, -2)
        elif self.in_transpose:
            x = torch.transpose(x, -1, -2)
            x = self.conv(x)
            x = self.out(self.dropout(x))
        if self.pooling:
            x = self.pool(x)
        if self.out_transpose:
            x = torch.transpose(x, -1, -2)
        return x

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features),requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features),requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class rfft2(nn.Module):
    def __init__(self):
        super(rfft2, self).__init__()

    def forward(self, x):
        return torch.fft.rfft2(x)
       
class CLS(nn.Module):
    def __init__(self, in_size, out_size=100,acti='relu'):
        super(CLS, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_size, 1000),
                                   actfunc(acti),
                                   nn.Linear(1000, out_size), nn.Sigmoid())

    def forward(self, x):
        x = x[0]
        out = self.model(x).view([1,-1])
        return out    
    
class Encoderlayer(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, inputsize, outsize, dropout=0.2, modelsize=None,acti='relu', headnum=16, fourier=False, res=False):
        super(Encoderlayer, self).__init__()
        if modelsize is None:
            self.res = True
            modelsize = int(inputsize / headnum)
        else:
            self.res = False
        if modelsize * headnum == outsize:
            self.resout = True
        else:
            self.resout = False
        if fourier:
            self.att = rfft2()
        else:
            self.att = SelfAttention(inputsize, headnum, modelsize)
        self.Wz = nn.Sequential(nn.Linear(modelsize * headnum, outsize), actfunc(acti), nn.Linear(outsize, outsize))
        self.sublayer = [LayerNorm(inputsize), LayerNorm(modelsize * headnum)]
        self.dropout = clones(nn.Dropout(dropout), 3)

    def forward(self, x):
        if self.res:
            z = x + self.dropout[0](self.att(self.sublayer[0](x)))
        else:
            z = self.att(x)
        if self.resout:
            out = z + self.dropout[1](self.Wz(self.sublayer[1](z)))
        else:
            out = self.Wz(z)
        return out
    
class Transformer(nn.Module):
    def __init__(self, N=5, src_vocab=64,d_model=256, dropout=0.25, h=8,outsize=2,acti='relu'):
        super(Transformer, self).__init__()
        self.Embed = nn.Sequential(
            Conv1dtranspose(src_vocab, d_model,acti=acti, in_transpose=True, out_transpose=True, kernel_size=7, stride=5),
            nn.Flatten(0, 1))
        self.model = nn.Sequential(
            PositionalEncoding(d_model, dropout),
            Encoder(Encoderlayer(inputsize=d_model, outsize=d_model, headnum=h, dropout=dropout,acti=acti), N),
            CLS(d_model,out_size=outsize,acti=acti)
        )
        self.cls = (torch.ones([1, d_model]) * -1)

    def forward(self, x):
        x = self.Embed(x)
        x = torch.cat([self.cls, x], dim=0)
        x = self.model(x)
        return x
    
# Codes for ResNet were implemented with Tensorflow = 2.10.0
# ResNet 
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model

def identity_block(X, f, filters):
    
    F1, F2, F3 = filters

    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)

    # ShortCut
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, s = 2):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path 
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(F3, (1, 1), strides = (1,1))(X)
    X = BatchNormalization(axis = 3)(X)

    # ShortCut
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)    
    
    return X

def ResNet(input_shape = (41, 4, 1), classes = 2):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 2), strides = (2, 1))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 1))(X)
    
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    
    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512],s = 2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    
    # Linear classifier
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet')
    
    return model
 
    
    
