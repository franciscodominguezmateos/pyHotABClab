#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 13/07/2019
@author: Francisco Dominguez
'''
import sys
import datetime as dt
import glob
import os
from os import listdir
import pickle
import cv2
import dlib
import numpy as np

#ML
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, MaxPooling2D, BatchNormalization, LeakyReLU, Conv2D
from keras.optimizers import RMSprop
from keras import backend as K

from PyQt5 import QtGui,QtWidgets, QtCore
import pyHotDraw
from pyHotDraw.Core.Qt5.pyHStandardEditor import pyHStandardEditor

from pyHotDraw.Images.pyHImage import pyHImage
from pyHotDraw.Figures.pyHImageFigure import pyHImageFigure

                    
def setShape(pyHImg):
    imgi=pyHImg.getData()
    imgo=fsd.process(imgi)
    pyHImg.setData(imgo)
    return len(fsd.getDetections())
    
def getGridFigure(imdic):
    gfusr=pyHGridFigure()
    for k in sorted(imdic):
        im=imdic[k][0]
        ar=im.getAspectRatio()
        imf=pyHImageFigure(0,0,40*ar,40,im)
        gfusr.addFigure(imf)    
    return gfusr

def isPortrait(pyHImg):
    return pyHImg.getWidth() < pyHImg.getHeight()
def getPortraitImageFigure(pyHImg,height=40):
    ar=pyHImg.getAspectRatio()
    imf=pyHImageFigure(0,0,height*ar,height,pyHImg)
    return imf
def getLandscapeImageFigure(pyHImg,width=40):
    ar=pyHImg.getAspectRatio()
    imf=pyHImageFigure(0,0,width,width/ar,pyHImg)
    return imf
def bestFitImageFigure(pyHImg,size=40):
    if isPortrait(pyHImg):
        return getPortraitImageFigure(pyHImg, size)
    return getLandscapeImageFigure(pyHImg, size)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = y_pred #K.square(y_pred)
    margin_square = K.maximum(margin - y_pred, 0) #K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_pairs_49(x,digit_indices):
    pairs =[]
    labels=[]
    di4=digit_indices[4]
    di9=digit_indices[9]
    n = min([len(di4),len(di9)]) - 1
    print("n=",n)
    #n = 200
    #4
    for i in range(n):
        di4a=di4[i]
        di4b=di4[i+1]
        pairs+=[[x[di4a],x[di4b]]]
        di9b=di9[i]
        pairs+=[[x[di4a],x[di9b]]]
        labels+=[1,0]
    #9
    for i in range(n):
        di9a=di9[i]
        di9b=di9[i+1]
        pairs+=[[x[di9a],x[di9b]]]
        di4b=di4[i]
        pairs+=[[x[di9a],x[di4b]]]
        labels+=[1,0]
    return np.array(pairs),np.array(labels)


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
input_shape = (img_rows, img_cols,1)
#Preprocess data
#With images and Conv2D we do not need to flatten
x_train = x_train.reshape(60000, img_rows, img_cols,1)
x_test  = x_test.reshape (10000, img_rows, img_cols,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
             
# create training+test positive and negative pairs
classes=range(10)
#digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
digit_indices = [np.where(y_train == i)[0] for i in classes]
for i,di in enumerate(digit_indices):
    print(i,len(di))
print ("-----")
tr_pairs, tr_y = create_pairs_49(x_train, digit_indices)
print(tr_pairs.shape)

#digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
digit_indices = [np.where(y_test == i)[0] for i in classes]
te_pairs, te_y = create_pairs_49(x_test, digit_indices)





def create_base_network(input_shape):
    ''' Base network to be shared (eq. to feature extraction) '''
    input = Input(shape=input_shape)
    x = Conv2D(15, (5,5), padding='same',input_shape=input_shape)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.5)(x)

    x = Conv2D(30, (9,9),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.5)(x)

    x = Flatten()(x)
    
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Dropout(0.5)(x)
    #for 2D view
    x = Dense(  2)(x)
    return Model(input, x)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

class pyHCVContractiveLossViewer(pyHStandardEditor):
    def __init__(self):
        super(pyHCVContractiveLossViewer, self).__init__()
        self.setWindowTitle('pyHCVContractiveLossViewer - Keras'+keras.__version__) 
        d=self.getView().getDrawing()
 
# train
batch_size=1000
epochs = 1
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))       
        
                                           
def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = pyHCVContractiveLossViewer()
    ex.timeElapsed=dt.datetime.now()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()        