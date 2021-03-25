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
import cv2
import dlib
import numpy as np
from PyQt5 import QtGui,QtWidgets, QtCore
from pyHotDraw.Core.Qt5.pyHStandardView import pyHStandardView
from pyHotDraw.Core.pyHAbstractEditor import pyHAbstractEditor
from pyHotDraw.Tools.pyHSelectionTool import pyHSelectionTool
from pyHotDraw.Figures.pyHPolylineFigure import pyHPolylineFigure
from pyHotDraw.Figures.pyHSplineFigure import pyHSplineFigure
from pyHotDraw.Figures.pyHEllipseFigure import pyHEllipseFigure
from pyHotDraw.Figures.pyHArcFigure import pyHArcFigure
from pyHotDraw.Geom.pyHPoint import pyHPoint
from pyHotDraw.Visitors.pyHGcodeGenerator import pyHGcodeGenerator
from pyHotDraw.Visitors.pyHPLTGenerator import pyHPLTGenerator
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

# given a path_base returns a dictionary with images where key is user number
# for SONY subfolders
def getUserID(f):
    #take away extension
    f=f[:-4]
    parts=f.split("_")
    return parts[1]

fsd=FaceShapeDetection()
def getpyHImagesFromFolder(path_base,ext="JPG",size=0.125):
    images={}
    for f in sorted(listdir(path_base))[:]:
        fext=f[-3:]
        if ext == fext:
            file_jpg=os.path.join(path_base,f)
            uid=getUserID(f)
            #print uid,file_jpg
            pyHImg=pyHImage(file_jpg,size)
            imgi=pyHImg.getData()
            imgo=fsd.process(imgi)
            images[uid]=(pyHImg,fsd.getDetections(),fsd.getShapes(),fsd.getDescriptors())
    return images
     
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
    return pyHImg.getWidth()<pyHImg.getHeight()
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

class pyHCVEditor(QtWidgets.QMainWindow,pyHAbstractEditor):
    def __init__(self):
        super(pyHCVEditor, self).__init__()
        pyHAbstractEditor.__init__(self)
        self.initActionMenuToolBars()
        self.statusBar()        
        self.initUI()
        d=self.getView().getDrawing()
               
        #from os import listdir
#         for r, dr, f in os.walk(path_base):
#             for file in f:
#                 if ".JPG" in file:
#                     print(os.path.join(r, file))
        path_base="/media/francisco/FREEDOS/datasets/pictures/FRAV-Attack/SONY/crop/from"
    
        # allImgs is a dic with attack key but users key is not a attack
        allImgs={}
        for dr in listdir(path_base):
            print dr
            dirpath=os.path.join(path_base,dr)
            if os.path.isdir(dirpath):
                allImgs[dr]=getpyHImagesFromFolder(dirpath)
        #imdic=getpyHImagesFromFolder(os.path.join(path_base,"users"))
        imdic=allImgs["users"]

        #gfusr=getGridFigure(imdic)
        #d.addFigure(gfusr)

        for attack in sorted(allImgs):
            if attack=="users":
                continue
            #attack="attack_01"
            print attack
            #gfusr=getGridFigure(allImgs[attack])
            #d.addFigure(gfusr)
            
            dic={}
            for usra in sorted(allImgs[attack]):
                min=1e20
                mid="NONE"
                for usr in imdic:
                    if len(imdic[usr][3])==0:
                        continue
                    if len(allImgs[attack][usra][3])==0:
                        #print usra
                        continue
                    usr_desc=imdic[usr][3][0]
                    att_desc=allImgs[attack][usra][3][0]
                    dif=usr_desc-att_desc
                    n=np.linalg.norm(np.array(dif))
                    if n<min:
                        min=n
                        mid=usr
                dic[usra]=mid
            gs=pyHGridFigure(cols=40)
            for usr in sorted(allImgs[attack]):
                im=allImgs[attack][usr][0]
                imf=getPortraitImageFigure(im)
                gs.addFigure(imf)
                if len(allImgs[attack][usr][3])==0:
                    print usr,"---"
                    gs.addFigure(getLandscapeImageFigure(pyHImage()))
                else:
                    im=imdic[dic[usr]][0]
                    imf=getPortraitImageFigure(im)
                    if dic[usr]!=usr:
                        print usr,dic[usr]
                        c=pyHAttributeColor(255,0,0)
                        imf.setColor(c)
                        w=pyHAttributeWidth(10)
                        imf.setAttribute("WIDTH",w)
                    gs.addFigure(imf)
            d.addFigure(gs)
                
#         gf=pyHGridFigure(rows=25,cols=8,cellWidth=40*6,cellHeight=40)
#         for k in sorted(imdic):
#             gfa=pyHGridFigure(rows=1,cols=6)
#             im=imdic[k]
#             setShape(im)
#             ar=im.getAspectRatio()
#             gfa.addFigure(pyHImageFigure(0,0,int(40*ar),40,im))
#             for a in sorted(allImgs):
#                 if a!="users":
#                     if k in allImgs[a]:
#                         im=allImgs[a][k]
#                         ar=im.getAspectRatio()
#                         imf=pyHImageFigure(0,0,int(40*ar),40,im)
#                         if setShape(im)==0:
#                             print k,a
#                             #pyhimg=pyHImageFigure(0,800,im.getWidth(),im.getHeight(),im)
#                             #d.addFigure(pyhimg)
#                             #im=pyHImage()
#                             c=pyHAttributeColor(255,0,0)
#                             imf.setColor(c)
#                             w=pyHAttributeWidth(4)
#                             imf.setAttribute("WIDTH",w)
#                         gfa.addFigure(imf)
#             gf.addFigure(gfa)
#         d.addFigure(gf)
                
#         for dr in listdir(path_base):
#             if dr=="attack_05":
#                 continue
#             dirpath=os.path.join(path_base,dr)
#             if os.path.isdir(dirpath):
#                 gf=pyHGridFigure()
#                 d.addFigure(gf)
#                 file_jpg=os.path.join(path_base,dr,"*.JPG")
#                 for filepath in sorted(glob.glob(file_jpg)):
#                     print("Processing file: {}".format(filepath)) 
#                     gf.addFigure(pyHImageFigure(0,0,40,40,pyHImage(filepath,0.125)))
        
#         path_base="/media/francisco/FREEDOS/datasets/pictures/FRAV-Attack/SONY/crop_face/front"
#         for dr in listdir(path_base):
#             dirpath=os.path.join(path_base,dr)
#             if os.path.isdir(dirpath):
#                 gf=pyHGridFigure()
#                 d.addFigure(gf)
#                 file_jpg=os.path.join(path_base,dr,"*.JPG")
#                 for filepath in sorted(glob.glob(file_jpg)):
#                     print("Processing file: {}".format(filepath)) 
#                     gf.addFigure(pyHImageFigure(0,0,40,40,pyHImage(filepath)))
                    
#         path_base="/media/francisco/FREEDOS/datasets/pictures/FRAV-Attack/RS/NIR/faces_NIR"
#         for dr in listdir(path_base)[:10]:
#             dirpath_users=os.path.join(path_base,dr)
#             if os.path.isdir(dirpath_users):
#                 gfu=pyHGridFigure(0,0,5,1,40*20)
#                 for dru in listdir(dirpath_users):
#                     dirpath=os.path.join(dirpath_users,dru)
#                     if os.path.isdir(dirpath):
#                         gf=pyHGridFigure(0,0,1,20)
#                         #d.addFigure(gf)
#                         file_jpg=os.path.join(dirpath,"*.jpg")
#                         for filepath in sorted(glob.glob(file_jpg))[:20]:
#                             print("Processing file: {}".format(filepath)) 
#                             gf.addFigure(pyHImageFigure(0,0,40,40,pyHImage(filepath,0.125)))
#                         gfu.addFigure(gf)
#                 d.addFigure(gfu)
                           
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
    def generateCode(self):
        item=self.treeWidget.currentItem()
        f=item.data(3,QtCore.Qt.ItemDataRole.UserRole)
        gc=pyHGcodeGenerator()
        sgc=f.visit(gc)
        self.gCodeEditor.setPlainText(sgc)
    def generatePLT(self):
        item=self.treeWidget.currentItem()
        f=item.data(3,QtCore.Qt.ItemDataRole.UserRole)
        gc=pyHPLTGenerator()
        sgc=f.visit(gc)
        self.gCodeEditor.setPlainText(sgc)
    def initUI(self):                       
        self.setView(pyHStandardView(self))
        
#         scrollArea = QtWidgets.QScrollArea()
#         scrollArea.setBackgroundRole(QtWidgets.QPalette.Dark)
#         scrollArea.setWidget(self.getView())
        
        self.setCentralWidget(self.getView())
        self.setGeometry(300, 30,900,500)
        self.setWindowTitle('pyHotVision Face Tool '+cv2.__version__)    
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