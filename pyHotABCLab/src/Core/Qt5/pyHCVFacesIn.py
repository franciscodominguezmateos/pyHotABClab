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
#for sklearn >0.17
#from sklearn.model_selection import train_test_split
#for sklearn <=0.17
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

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


fsd=FaceShapeDetection()
                    
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

''' FacialExpressions dataset utility functions '''
path_base="/media/francisco/FREEDOS/datasets/pictures/facial_expressions"
def getpyHImagesFromFacialExpressions(path_base):
    file_csv=path_base+"/data/legend.csv"
    print("Processing file: {}".format(file_csv)) 
    with open(file_csv) as fp:  
        # avoid header
        line = fp.readline()
        line = fp.readline() #this doesn't seem to be a string?!?!?
        cnt = 1
        faces={}
        expressions=[]
        while line:
            line_list=line.split(",")
            _,file_name,expression=line.split(",")
            if file_name=="Abdoulaye_Wade_0001.jpg":
                print "stop"
            # remove \n and make lowercase
            expression=expression.rstrip().lower()
            # file name format is name_surname_number
            # next line remove number
            name="_".join(file_name.split("_")[:-1])
                #print("Line {}: {} {} {} {}".format(cnt, line_list[0], line_list[1], line_list[2],name))
            full_file_name=path_base+"/images/"+file_name
            exists = os.path.isfile(full_file_name)
            if not exists:
                print("File {} doesn't exist.".format(file_name))
                line = fp.readline()
                continue
            pyHImg=pyHImage(full_file_name)
            imgi=pyHImg.getData()
            imgo=fsd.process(imgi)
            if len(fsd.getDetections())==0:
                line = fp.readline()
                continue
            if not expression in expressions:
                expressions.append(expression)
            if not name in faces.keys():
                faces[name]={}
            if not expression in faces[name].keys():
                faces[name][expression]=[]
            faces[name][expression].append((full_file_name,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors()))
            line = fp.readline()
            cnt += 1
            print cnt,file_name
    c=0;
    expressions.sort()
    exp={}
    for i,e in enumerate(expressions):
        exp[e]=i
#     print (expressions)
#     print (exp)
#     for i,name in enumerate(sorted(faces.keys())):
#         if(len(faces[name])>1):
#             c+=1
#             print("{} {} {} {}".format(i,c,name,len(faces[name])))
    return faces,expressions,exp

def getOneHotEncodingExpression(pexpression):
#['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
#{'neutral': 5, 'sadness': 6, 'happiness': 4, 'disgust': 2, 'anger': 0, 'surprise': 7, 'fear': 3, 'contempt': 1}
    expression=pexpression.strip()
    if expression=="anger":
        return np.array([1,0,0,0,0,0,0,0])
    if expression=="contempt":
        return np.array([0,1,0,0,0,0,0,0])
    if expression=="disgust":
        return np.array([0,0,1,0,0,0,0,0])
    if expression=="fear":
        return np.array([0,0,0,1,0,0,0,0])
    if expression=="happiness":
        return np.array([0,0,0,0,1,0,0,0])
    if expression=="neutral":
        return np.array([0,0,0,0,0,1,0,0])
    if expression=="sadness":
        return np.array([0,0,0,0,0,0,1,0])
    if expression=="surprise":
        return np.array([0,0,0,0,0,0,0,1])
    print "el ="+expression
    return np.array([1,1,1,1,1,1,1])

def makeDataExpressions(dic):
    features=[]
    labels=[]
    for usr in sorted(dic):
        if len(dic[usr])<0:
            continue
        for expression in sorted(dic[usr]):
            if not expression in ["anger","disgust","fear","surprise","sadness","contempt"]:
                continue
            for pyHimg,detections,shapes,descriptors in dic[usr][expression]:  #this is a list of tuplas
                if len(descriptors)!=0:
                    features.append(descriptors[0]) #only add the firs face of the picture
                    labels.append(getOneHotEncodingExpression(expression))
    npf=np.array(features)
    for i in labels:
        if len(i)!=8:
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
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
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

class pyHCVEditor(QtWidgets.QMainWindow,pyHAbstractEditor):
    def __init__(self):
        super(pyHCVEditor, self).__init__()
        pyHAbstractEditor.__init__(self)
        self.initActionMenuToolBars()
        self.statusBar()        
        self.initUI()
        d=self.getView().getDrawing()
        
        path_base="/media/francisco/FREEDOS/datasets/pictures/facial_expressions"

        dump=0
        if dump==1:
            allFE=getpyHImagesFromFacialExpressions(path_base)
            pickle_file = file('allFE.p', 'w')
            pickle.dump(allFE,pickle_file)
        else:
            print "Loadidng data, please wait...."
            pickled_file = open('allFE.p')
            allFE = pickle.load(pickled_file)
        faces,expressions,exp=allFE
        print expressions
        print exp
        ef=pyHGridFigure(0,0,10,1,60,20)
        for e in expressions:
            tf=pyHTextFigure(0,0,60,20," "+e+" ")
            ef.addFigure(tf)
        d.addFigure(ef)
        gf=pyHGridFigure(0,0,60,100)
#         for uid in sorted(faces)[:]:
#             for expression in sorted(faces[uid])[:1]:
#                 if not expression in ["fear"]:
#                     continue
#                 file_name,detections,shapes,descriptors=faces[uid][expression][0]
#                 imgf=bestFitImageFigure(pyHImage(file_name))
#                 gf.addFigure(imgf)
        for uid in sorted(faces)[:100]:
            for expression in sorted(faces[uid])[:]:
                if not expression in ["neutral"]:
                    continue
                for file_name,detections,shapes,descriptors in faces[uid][expression]:
                    imgf=bestFitImageFigure(pyHImage(file_name))
                    gf.addFigure(imgf)
        for uid in sorted(faces)[:100]:
            print uid
            for expression in sorted(faces[uid])[:]:
                if not expression in ["happiness"]:
                    continue
                for file_name,detections,shapes,descriptors in faces[uid][expression]:
                    imgf=bestFitImageFigure(pyHImage(file_name))
                    imgf.setColor(0,255,0)
                    gf.addFigure(imgf)
        d.addFigure(gf)
#         gf=pyHGridFigure(0,0,4,6,440,240)
#         for uid in sorted(allFE):
#             cf=getConfusionMatrix(faces,uid)
#             gf.addFigure(cf)
#         d.addFigure(gf)
#         
        ''' MODEL '''
        x_train, x_test, y_train, y_test=makeDataExpressions(faces)
        print x_train.shape,x_test.shape,y_train.shape,y_test.shape
        print y_train[0],y_test[0]
        print y_train[1],y_test[1]
        print y_train[2],y_test[2]
        print y_train[3],y_test[3]
        fit=1
        if fit==1:
            model=makeModel()
            model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
            batch_size=5000
            epochs=1000
            history = model.fit(x_train, y_train,
                        batch_size=batch_size, nb_epoch=epochs,
                        verbose=1, validation_data=(x_test, y_test))
            model.save("allFE.h5")
        else:
            model=load_model("allFE.h5")
         
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        self.getView().setTransformFitToDrawing()

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
        self.setWindowTitle('pyHVBBCattack MLP. OpenCV '+cv2.__version__+", Keras "+keras.__version__)    
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
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()        