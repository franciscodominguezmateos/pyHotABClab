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
from scipy.stats import multivariate_normal
from sklearn import mixture

#ML
#for sklearn >0.17
from sklearn.model_selection import train_test_split
#for sklearn <=0.17
#from sklearn.cross_validation import train_test_split
#import keras
#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense, Dropout
#from keras.optimizers import RMSprop

from PyQt5 import QtGui,QtWidgets, QtCore
from pyHotDraw.Core.Qt5.pyHStandardView import pyHStandardView
from pyHotDraw.Core.pyHAbstractEditor import pyHAbstractEditor
from pyHotDraw.Tools.pyHSelectionTool import pyHSelectionTool
from pyHotDraw.Figures.pyHPolylineFigure import pyHPolylineFigure
from pyHotDraw.Figures.pyHSplineFigure import pyHSplineFigure
from pyHotDraw.Figures.pyHEllipseFigure import pyHEllipseFigure
from pyHotDraw.Figures.pyHArcFigure import pyHArcFigure
from pyHotDraw.Geom.pyHPoint import pyHPoint
from pyHotDraw.Images.pyHImage import pyHImage
from pyHotDraw.Figures.pyHAttributes import pyHAttributeColor,pyHAttributeWidth
from pyHotDraw.Figures.pyHImageFigure import pyHImageFigure
from pyHotDraw.Figures.pyHImageFigure import pyHImageDottedFigure
from pyHotDraw.Figures.pyHImageFigure import pyHImagesMixedFigure
from pyHotDraw.Figures.pyHImageFigure import pyHCameraFigure
from pyHotDraw.Figures.pyHImageFigure import pyHImageFilterFigure
from pyHotDraw.Figures.pyHImageFigure import pyHImageSecFilterFigure
from pyHotDraw.Figures.pyHGridFigure import pyHGridFigure
from pyHotDraw.Figures.pyHTextFigure import pyHTextFigure
from pyHotDraw.Images.pyHImageFilters import SobelX
from pyHotDraw.Images.pyHImageFilters import SobelY
from pyHotDraw.Images.pyHImageFilters import Gaussian
from pyHotDraw.Images.pyHImageFilters import OpticalFlow
from pyHotDraw.Images.pyHImageFilters import FeatureDetector
from pyHotDraw.Images.pyHImageFilters import FastFeatureDetector
from pyHotDraw.Images.pyHImageFilters import FlannMacher
from pyHotDraw.Images.pyHImageFilters import FundamentalMatrix
from pyHotDraw.Images.pyHImageFilters import HomographyMatrix
from pyHotDraw.Images.pyHImageFilters import HistogramColor
from pyHotDraw.Images.pyHImageFilters import FaceShapeDetection
from pyHotDraw.Figures.pyHCompositeFigure import pyHCompositeFigure

class ABCFace:
    def __init__(self):
        self.fileName=""
        self.filePath=""
        
class ABCFaceEmbedding(ABCFace):
    fsd=FaceShapeDetection()
    def __init__(self):
        self.descriptor=None
        self.shape=None
        self.detection=None
    def buildFromFile(self):
        file_jpg=os.path.join(path_base,f)
        #print file_jpg
        #print uid,file_jpg
        pyHImg=pyHImage(file_jpg,size)
        imgi=pyHImg.getData()
        imgo=fsd.process(imgi)
        if len(fsd.getDetections())==0:
            return
        self.descriptor=fsd.getDetections()
        self.shape     =fsd.getShapes()
        self.descriptor=fsd.getDescriptors()
        
class ABCAttack:
    def __init__(self,id):
        self.id=id
        self.faces=[]
    def buildFromFolder(self):
        path_base=os.path.joint(self.path_base,self.id)        
        images=[]
        path_base=self.path_base
        for f in sorted(listdir(path_base))[:]:
            fext=f[-3:]
            if ext == fext.lower():
                file_jpg=os.path.join(path_base,f)
                #print file_jpg
                #print uid,file_jpg
                pyHImg=pyHImage(file_jpg,size)
                imgi=pyHImg.getData()
                imgo=fsd.process(imgi)
                if len(fsd.getDetections())==0:
                    continue
                #images.append((pyHImg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
                #fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()
class ABCUser:
    def __init__(self,id):
        #id is the folder
        self.id=id
        self.attacks=[]
    def buildFromFolder(self):
        path_base=os.path.joint(self.path_base,self.id)
        for dr in sorted(listdir(path_base)):
            dirpath=os.path.join(path_base,dr)
            if os.path.isdir(dirpath):
                #print dr
                fl=getpyHImagesFromFolderRS(dirpath,ext,size)
                if len(fl)==0: continue
                usrImgs[dr]=fl
        
class ABCLab:
    path_base="/media/francisco/FREEDOS/datasets/pictures/FRAV-Attack/RS/NIR/faces_NIR"
    def __init__(self):
        self.users=[]
class ABCLabEmbedding(ABCLab):
    fsd=FaceShapeDetection()
    I=np.identity(128)
    def __init__(self):
        pass
    def getUserIdRS(f):
        #take away extension
        parts=f.split("_")
        return parts[0]
    def getpyHImagesFromFolderRS(self,ext="jpg",size=0.125):
        images=[]
        path_base=self.path_base
        for f in sorted(listdir(path_base))[:]:
            fext=f[-3:]
            if ext == fext.lower():
                file_jpg=os.path.join(path_base,f)
                #print file_jpg
                #print uid,file_jpg
                pyHImg=pyHImage(file_jpg,size)
                imgi=pyHImg.getData()
                imgo=fsd.process(imgi)
                if len(fsd.getDetections())==0:
                    continue
                #images.append((pyHImg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
                images.append((file_jpg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
        return images
    def getpyHImagesFromUserRS(self,ext="jpg",size=0.125):
        usrImgs={}
        path_base=self.path_base
        for dr in sorted(listdir(path_base)):
            dirpath=os.path.join(path_base,dr)
            if os.path.isdir(dirpath):
                #print dr
                fl=getpyHImagesFromFolderRS(dirpath,ext,size)
                if len(fl)==0: continue
                usrImgs[dr]=fl
        return usrImgs
    def getpyHImagesAllRS(self,ext='jpg',size=0.25):
        allImgs={}
        path_base=self.path_base
        for dr in listdir(path_base)[:20]:
            dirpath=os.path.join(path_base,dr)
            if os.path.isdir(dirpath):
                print dr
                allImgs[dr]=getpyHImagesFromUserRS(dirpath,ext,size)
        return allImgs
        

fsd=FaceShapeDetection()
I=np.identity(128)
                    
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

''' RS utility functions '''
def getOneHotEncodingAttackRS(pattack):
    attack=pattack.strip()
    if attack=="user":
        return np.array([1,0,0,0,0,0])
    if attack=="attack_01":
        return np.array([0,1,0,0,0,0])
    if attack=="attack_02":
        return np.array([0,0,1,0,0,0])
    if attack=="attack_03":
        return np.array([0,0,0,1,0,0])
    if attack=="attack_04":
        return np.array([0,0,0,0,1,0])
    if attack=="attack_05":
        return np.array([0,0,0,0,0,1])
    print "el ="+attack
    return np.array([1,1,1,1,1,1,1,1,1,1])
def getUserIdRS(f):
    #take away extension
    parts=f.split("_")
    return parts[0]
def getpyHImagesFromFolderRS(path_base,ext="jpg",size=0.125):
    images=[]
    for f in sorted(listdir(path_base))[:]:
        fext=f[-3:]
        if ext == fext.lower():
            file_jpg=os.path.join(path_base,f)
            #print file_jpg
            #print uid,file_jpg
            pyHImg=pyHImage(file_jpg,size)
            imgi=pyHImg.getData()
            imgo=fsd.process(imgi)
            if len(fsd.getDetections())==0:
                continue
            #images.append((pyHImg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
            images.append((file_jpg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
    return images
def getpyHImagesFromUserRS(path_base,ext="jpg",size=0.125):
    usrImgs={}
    for dr in sorted(listdir(path_base)):
        dirpath=os.path.join(path_base,dr)
        if os.path.isdir(dirpath):
            #print dr
            fl=getpyHImagesFromFolderRS(dirpath,ext,size)
            if len(fl)==0: continue
            usrImgs[dr]=fl
    return usrImgs

def getpyHImagesAllRS(path_base,ext='jpg',size=0.25):
    allImgs={}
    for dr in listdir(path_base)[:2]:
        dirpath=os.path.join(path_base,dr)
        if os.path.isdir(dirpath):
            print dr
            allImgs[dr]=getpyHImagesFromUserRS(dirpath,ext,size)
    return allImgs

def makeDataRS(dic):
    features=[]
    labels=[]
    for usr in sorted(dic):
        for attack in sorted(dic[usr]):
            if attack in ["attack_010","attack_010","attack_040","attack_050"]:
                continue   
            #print usr,attack,len(dic[usr][attack])
            for pyHimg,detections,shapes,descriptors in dic[usr][attack]:  #this is a list of tuplas
                if len(descriptors)!=0:
                    features.append(descriptors[0]) #only add the firs face of the picture
                    labels.append(getOneHotEncodingAttackRS(attack))
    npf=np.array(features)
    for i in labels:
        if len(i)!=6:
            print "ERROR",len(i)
    #print len(labels),len(labels[0])
    npl=np.array(labels)
    #print npf.shape,npl.shape,npl[0]
    # Use `train_test_split` here.
    X_train, X_val, y_train, y_val = train_test_split(npf,npl, random_state=0, test_size=0.1)
    return X_train, X_val, y_train, y_val

def makeModel():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(128,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.summary()
    return model

def npDescriptors(ld):
    l=[]
    for pyHimg,detections,shapes,descriptors in ld:
        for d in descriptors:
            l.append(d)
    npl=np.array(l)
    #print npl.shape
    return npl

def meanDescriptors(ld):
    npl=npDescriptors(ld)
    #print npl.shape
    mean=np.mean(npl,axis=0)
    #print mean.shape
    return mean
def sdevDescriptors(ld):
    npl=npDescriptors(ld)
    #print npl.shape
    sdev=np.std(npl,axis=0)
    #print mean.shape
    return sdev
#Compute Bayesian Gaussian Mixture Model or Bayesian Mixture of Gaussians
#There are two versions Dirichlet distribution prior and dirichlet process prior 
# the second seem to work better
#https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
def bmog(l):
    n_components=3
    random_state=2
    #npl=npDescriptors(ld)
    npl=np.array(l)
    estimator=mixture.BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components= n_components, reg_covar=0.0001, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state)
    estimator.weight_concentration_prior = 10000
    estimator.fit(npl)
    return estimator
    
#https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
#There's no closed form for the KL divergence between GMMs. You can easily do Monte Carlo, though. Recall that KL(p||q) = \int p(x) log(p(x) / q(x)) dx = E_p[ log(p(x) / q(x)). So:

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X,y = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

#(mean(log(p(x) / q(x))) = mean(log(p(x)) - log(q(x))) = mean(log(p(x))) - mean(log(q(x))) is somewhat cheaper computationally.)

#You don't want to use scipy.stats.entropy; that's for discrete distributions.

#If you want the symmetrized and smoothed Jensen-Shannon divergence KL(p||(p+q)/2) + KL(q||(p+q)/2) instead, it's pretty similar:

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)
    log_p_X, _ = gmm_p.score_samples(X)
    log_q_X, _ = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)
    log_p_Y, _ = gmm_p.score_samples(Y)
    log_q_Y, _ = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (  log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

#(log_mix_X/log_mix_Y are actually the log of twice the mixture densities; pulling that out of the mean operation saves some flops.)

def asLists(dataImages):
    pyHimgs=[]
    detections=[]
    shapes=[]
    descriptors=[]
    #These are actually lists because in an image can be more than one face but this is not the case
    #each image has only a face
    for pyImg,detection,shape,descriptor in dataImages:
        pyHimgs.append(pyImg)
        detections.append(np.array(detection[0]))
        shapes.append(np.array(shape[0]))
        descriptors.append(np.array(descriptor[0]))
    return pyHimgs,detections,shapes,descriptors

def getConfusionMatrixKL(allRS,uid):
    attBMoG={}
    attDesc={}
    cf=pyHCompositeFigure()
    fid=pyHTextFigure(0,200,40,40,uid[5:])
    ''' HEADs '''
    gfh=pyHGridFigure( 40, 200, 1,10)
    gfv=pyHGridFigure(  0,   0, 5, 1)
    ''' means '''
    for i,att in enumerate(sorted(allRS[uid])):
        if len(allRS[uid][att])==0:
            continue
        pyHimg,detections,shapes,descriptors=asLists(allRS[uid][att])
        if len(descriptors)<4:
            print("Less than 2 images uid=",uid,"attack=",att) 
            continue
        attBMoG[att]=bmog(descriptors[1:])
        imgfh=bestFitImageFigure(pyHImage(pyHimg[0]))
        imgfv=bestFitImageFigure(pyHImage(pyHimg[0]))
        imgfh.setColor(0,255,0)
        imgfv.setColor(0,255,0)
        imgfh.setWidth(5)
        imgfv.setWidth(5)
        gfh.addFigure(imgfh)
        gfv.addFigure(imgfv)
    ''' sample 1 '''
    for i,att in enumerate(sorted(allRS[uid])):
        if len(allRS[uid][att])==0:
            continue
        pyHimg,detections,shapes,descriptors=asLists(allRS[uid][att])
        if len(descriptors)<2:
            print("Less than 2 images uid=",uid,"attack=",att) 
            continue
        #print att,len(allRS[uid][att]),len(detections),len(descriptors)
        attDesc[att]=descriptors[0]
        imgfh=bestFitImageFigure(pyHImage(pyHimg[1]))
        gfh.addFigure(imgfh)
    cf.addFigure(gfv)        
    cf.addFigure(gfh)
    ''' BODY '''
    pmf=pyHGridFigure(40,0,5,10,40)
    for i,bmogH in enumerate(sorted(attBMoG)):
        for j,bmogV in enumerate(sorted(attBMoG)):
            dif=gmm_kl(attBMoG[bmogH],attBMoG[bmogV])
            #dif=multivariate_normal.pdf(attMean[a1],attMean[a0],attSdev[a0])
            #print "dif=",dif
            print uid,":   KL:",bmogH," -> ",bmogV,"=",dif
            tf=pyHTextFigure(0,0,40,40,dif)
            if j>4 and i<5: tf.setFillColor(0, 0, 255, 100)
            if j>4 and i>=5: tf.setFillColor(255, 0, 255, 100)
            if dif<0.1: tf.setFillColor(255, 0, 0, 100)
            pmf.addFigure(tf)
        for j,bmogV in enumerate(sorted(attBMoG)):
            #score=attBMoG[bmogV].score([attDesc[bmogH]])
            score=attBMoG[bmogV].score([attDesc[bmogH]])
            print uid,":Score:",bmogH," -> ",bmogV,"=",score
            tf=pyHTextFigure(0,0,40,40,score)
            if j>4 and i<5: tf.setFillColor(0, 0, 255, 100)
            if j>4 and i>=5: tf.setFillColor(255, 0, 255, 100)
            if abs(score)<500: tf.setFillColor(255, 0, 0, 100)
            pmf.addFigure(tf)
    cf.addFigure(pmf)
    cf.addFigure(fid)
    print
    return cf

def getConfusionMatrix(allRS,uid):
    attMean={}
    attSdev={}
    
    cf=pyHCompositeFigure()
    fid=pyHTextFigure(0,200,40,40,uid[5:])
    ''' HEADs '''
    gfh=pyHGridFigure( 40, 200, 1,10)
    gfv=pyHGridFigure(  0,   0, 5, 1)
    ''' means '''
    for i,att in enumerate(sorted(allRS[uid])):
        if len(allRS[uid][att])==0:
            continue
        pyHimg,detections,shapes,descriptors=allRS[uid][att][0]
        #if len(descriptors)<2: continue
        attMean[att]=meanDescriptors(allRS[uid][att])
        attSdev[att]=sdevDescriptors(allRS[uid][att])
        imgfh=bestFitImageFigure(pyHImage(pyHimg))
        imgfv=bestFitImageFigure(pyHImage(pyHimg))
        imgfh.setColor(0,255,0)
        imgfv.setColor(0,255,0)
        imgfh.setWidth(5)
        imgfv.setWidth(5)
        gfh.addFigure(imgfh)
        gfv.addFigure(imgfv)
    ''' sample 1 '''
    for i,att in enumerate(sorted(allRS[uid])):
        if len(allRS[uid][att])==0:
            continue
        pyHimg,detections,shapes,descriptors=allRS[uid][att][0]
        #if len(descriptors)<2: continue
        #print att,len(allRS[uid][att]),len(detections),len(descriptors)
        attMean["z"+att]=descriptors[0]
        imgfh=bestFitImageFigure(pyHImage(pyHimg))
        gfh.addFigure(imgfh)
    cf.addFigure(gfv)        
    cf.addFigure(gfh)
    ''' BODY '''
    pmf=pyHGridFigure(40,0,5,10,40)
    for i,a0 in enumerate(sorted(attMean)[:5]):
        threshold=np.linalg.norm(attSdev[a0])
        print uid,a0,"std:",threshold
        for j,a1 in enumerate(sorted(attMean)):
            print "    -",a1
            dif=np.linalg.norm(attMean[a0]-attMean[a1])-threshold
            #dif=multivariate_normal.pdf(attMean[a1],attMean[a0],attSdev[a0])
            #print "dif=",dif
            tf=pyHTextFigure(0,0,40,40,dif)
            if j>4 and i<5: tf.setFillColor(0, 0, 255, 100)
            if j>4 and i>=5: tf.setFillColor(255, 0, 255, 100)
            if dif<0.1: tf.setFillColor(255, 0, 0, 100)
            pmf.addFigure(tf)
    cf.addFigure(pmf)
    cf.addFigure(fid)
    print
    return cf

class pyHCVEditor(QtWidgets.QMainWindow,pyHAbstractEditor):
    def __init__(self):
        super(pyHCVEditor, self).__init__()
        pyHAbstractEditor.__init__(self)
        self.initActionMenuToolBars()
        self.statusBar()        
        self.initUI()
        d=self.getView().getDrawing()
        
        path_base="/home/francisco/datasets/datasets/pictures/FRAV-Attack/RS/NIR/faces_NIR"
        dump=1
        if dump==1:
            allRS=getpyHImagesAllRS(path_base,"jpg")
            pickle_file = file('20RS.p', 'w')
            pickle.dump(allRS,pickle_file)
        else:
            print "Loadidng data, please wait...."
            pickled_file = open('20RS.p')
            #pickled_file = open('allRS.p')
            allRS = pickle.load(pickled_file)
            print "Loaded data, thank you."
            
        gf=pyHGridFigure(0,0,23,6,440,240)
        #gf=pyHGridFigure(0,0,10,12,440,240)
        for uid in sorted(allRS)[:]:
            if len(allRS[uid])!=5:
               continue
            cf=getConfusionMatrixKL(allRS,uid)
            gf.addFigure(cf)
        d.addFigure(gf)
        
        ''' MODEL '''
#         x_train, x_test, y_train, y_test=makeDataRS(allRS)
#         fit=1
#         if fit==1:
#             model=makeModel()
#             model.compile(loss='categorical_crossentropy',
#                   optimizer=RMSprop(),
#                   metrics=['accuracy'])
#             batch_size=150
#             epochs=50
#             history = model.fit(x_train, y_train,
#                         batch_size=batch_size, nb_epoch=epochs,
#                         verbose=1, validation_data=(x_test, y_test))
#             model.save("allRS.h5")
#         else:
#             model=load_model("allRS.h5")
#         
#         score = model.evaluate(x_test, y_test, verbose=0)
#         print('Test loss:', score[0])
#         print('Test accuracy:', score[1])
#         
#         self.getView().setTransformFitToDrawing()

#Redefinning abstract methods
    def createMenuBar(self):
        return QtWidgets.QMainWindow.menuBar(self)
    def addMenuAndToolBar(self,name):
        self.menu[name]=self.menuBar.addMenu(name)
        self.toolBar[name]=self.addToolBar(name)
        self.actionGroup[name]=QtWidgets.QActionGroup(self)
    def addAction(self,menuName,icon,name,container,sortCut,statusTip,connect,addToActionGroup=False):
        a=QtWidgets.QAction(QtGui.QIcon(icon),name,container)
        a.setObjectName(name)
        a.setShortcut(sortCut)
        a.setStatusTip(statusTip)
        a.triggered.connect(connect)
        if addToActionGroup:
            a.setCheckable(True)
            self.actionGroup[menuName].addAction(a)
        self.menu[menuName].addAction(a)
        self.toolBar[menuName].addAction(a)
        #print "a.objectName:"+a.objectName()
        
    def initActionMenuToolBars(self):
        self.addMenuAndToolBar("&File")
        self.addAction("&File","../images/fileNew.png",'New',self,"Ctrl+N","New document",self.newFile)
        self.addAction("&File","../images/fileOpen.png",'Open',self,"Ctrl+O","Open document",self.openFile)
        self.addAction("&File","../images/fileSave.png",'Save',self,"Ctrl+Q","Save document",self.selectingFigures)
        self.addAction("&File","",'Exit',self,"Ctrl+Q","Exit application",self.close)
        self.addMenuAndToolBar("&Edit")
        self.addAction("&Edit","../images/editCopy.png",'Copy',self,"Ctrl+C","Copy",self.copy)
        self.addAction("&Edit","../images/editCut.png",'Cut',self,"Ctrl+X","Cut",self.cut)
        self.addAction("&Edit","../images/editPaste.png",'Paste',self,"Ctrl+V","Paste",self.paste)
        self.addAction("&Edit","../images/editUndo.png",'Paste',self,"Ctrl+V","Paste",self.selectingFigures)
        self.addAction("&Edit","../images/editRedo.png",'Paste',self,"Ctrl+V","Paste",self.selectingFigures)
        dbUnits=QtWidgets.QComboBox(self)
        dbUnits.addItem("Milimetros")
        dbUnits.addItem("Pulgadas")
        dbUnits.addItem("Pixels")
        dbUnits.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        # create a menu item for our context menu.
        a = QtWidgets.QAction("A try...", self)
        dbUnits.addAction(a)
        a = QtWidgets.QAction("A try...", self)
        dbUnits.addAction(a)
        a = QtWidgets.QAction("A try...", self)
        dbUnits.addAction(a)
        self.toolBar["&Edit"].addWidget(dbUnits)
        self.addAction("&Edit","../images/zoom.png",'Zoom',self,"Ctrl+V","Zoom",self.selectingFigures)
        sceneScaleCombo = QtWidgets.QComboBox()
        sceneScaleCombo.addItems(["50%", "75%", "100%", "125%", "150%", "200%", "250%", "300%", "350%", "400%"])
        sceneScaleCombo.setCurrentIndex(2)
        sceneScaleCombo.setEditable(True)
        sceneScaleCombo.currentIndexChanged.connect(self.onScaleChanged)
        self.toolBar["&Edit"].addWidget(sceneScaleCombo)
        self.addMenuAndToolBar("&CAD")
        self.addAction("&CAD","../images/selectionTool.png",'Selection',self,"Ctrl+S","Selection Tool",self.selectingFigures,True)
        self.addAction("&CAD","../images/move.png",'View trasnlate',self,"Ctrl+v","View Translate Tool",self.viewTranslate,True)
        self.addAction("&CAD","../images/bug.png",'Camera',self,"Ctrl+S","Create Camera",self.creatingCamera,True)
        self.addAction("&CAD","../images/createRoundRectangle.png",'Create Image',self,"Ctrl+S","Selection Tool",self.creatingImage,True)
        self.addAction("&CAD","../images/createLineConnection.png",'Create Image Filter connection',self,"Ctrl+S","Create Image Filter connection Tool",self.creatingLineImageFilterConnection,True)
        self.addAction("&CAD","../images/createLineConnection.png",'Create connection',self,"Ctrl+S","Create connection Tool",self.creatingLineConnection,True)
        self.addAction("&CAD","../images/createPolygon.png",'Polyline',self,"Ctrl+S","Creatting Polyline",self.creatingPolyline,True)
        self.addAction("&CAD","../images/createLine.png",'Line',self,"Ctrl+S","Selection Tool",self.creatingLine,True)
        self.addAction("&CAD","../images/createRectangle.png",'Rectangle',self,"Ctrl+S","Create Rectangle Tool",self.creatingRectangle,True)
        self.addAction("&CAD","../images/createRoundRectangle.png",'Round Rectangle',self,"Ctrl+S","Selection Tool",self.creatingRectangle,True)
        self.addAction("&CAD","../images/createEllipse.png",'Ellipse',self,"Ctrl+S","Selection Tool",self.creatingEllipse,True)
        self.addAction("&CAD","../images/createDiamond.png",'Ellipse',self,"Ctrl+S","Selection Tool",self.creatingDiamond,True)
        self.addAction("&CAD","../images/createScribble.png",'Spline',self,"Ctrl+S","Spline Tool",self.creatingSpline,True)
        self.addAction("&CAD","../images/jointPoints1.png",'Join',self,"Ctrl+S","Join points",self.join,False)
        self.addAction("&CAD","../images/selectionGroup.png",'Selection Group',self,"Ctrl+S","Selection Group",self.selectionGroup,False)
        self.addAction("&CAD","../images/selectionUngroup.png",'Selection Ungroup',self,"Ctrl+S","Selection Ungroup",self.selectionUngroup,False)
        self.addAction("&CAD","../images/moveToBack.png",'Move to Back',self,"Ctrl+S","Move to Back",self.moveBack,False)
        self.addAction("&CAD","../images/moveToFront.png",'Move to Front',self,"Ctrl+S","Move to Front",self.moveFront,False)

    def onScaleChanged(self,index):
        s=float(index)
        t=self.getView().getTransform()
        t.sx=s+0.50
        t.sy=s+0.50
        self.getView().update()
    def newFile(self):
        self.getView().getDrawing().clearFigures()
        self.getView().update()
    def openFile(self):
        self.getView().getDrawing().clearFigures()
        fileNames = QtWidgets.QFileDialog.getOpenFileName(self,("Open Image"), QtCore.QDir.currentPath(), ("Image Files (*.dxf)"))
        if not fileNames:
            fileName="C:\\Users\\paco\\Desktop\\a4x2laser.dxf"
        else:
            fileName=fileNames[0]
        self.openDXF(fileName)
        self.getView().update()
        
    def updateDraw(self,item,col):
        print "item changed "+str(col)+"="+item.data(col,QtCore.Qt.DisplayRole)+" "+item.data(3,QtCore.Qt.ItemDataRole.UserRole).__class__.__name__
    def initUI(self):                       
        self.setView(pyHStandardView(self))
        
#         scrollArea = QtWidgets.QScrollArea()
#         scrollArea.setBackgroundRole(QtWidgets.QPalette.Dark)
#         scrollArea.setWidget(self.getView())
        
        self.setCentralWidget(self.getView())
        self.setGeometry(300, 30,900,500)
        #self.setWindowTitle('pyHVBBCattack MLP. OpenCV '+cv2.__version__+", Keras "+keras.__version__)    
        self.setWindowTitle('pyHVBBCattack MLP. OpenCV '+cv2.__version__)    
        self.sb=QtWidgets.QLabel(self)
        self.sb.setText("x=0,y=0")
        self.statusBar().addPermanentWidget(self.sb)
        self.sb1=QtWidgets.QLabel(self)
        self.setCurrentTool(pyHSelectionTool(self.getView()))
        self.show()
                                   
def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = pyHCVEditor()
    
    ex.timeElapsed=dt.datetime.now()
    #ex.openDXF("a4x2laser.dxf")
    #puertos_disponibles=scan(num_ports=20,verbose=True)
    #ex.openPLT("a4x2laser.plt")
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()        