# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:52:00 2020

@author: aylin
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMainWindow, QMessageBox, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import pandas as pd
import openpyxl
from uygulama import Ui_Dialog
from sklearn import preprocessing
import seaborn as sns
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import genfromtxt
from PyQt5.QtCore import QAbstractTableModel, Qt
import random
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC as svc
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os,shutil
import cv2
#from skimage import io,color
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
#from keras import Sequential
#from keras.layers import Conv2D, Flatten, Dense, Dropout
from glob import glob
import scikitplot.metrics as splt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from os.path import expanduser
import imutils

class MainWindow(QWidget,Ui_Dialog):
    dataset_file_path = ""
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.importet)
        self.pushButton_2.clicked.connect(self.ikincisayfayagec)
        self.pushButton_5.clicked.connect(self.birincisayfayadon)
        self.pushButton_8.clicked.connect(self.ucuncusayfayagec)
        self.pushButton_9.clicked.connect(self.ikincisayfayadon)
        self.pushButton_4.clicked.connect(self.renkuzaylari)
        self.pushButton_3.clicked.connect(self.uygula)
        self.pushButton_6.clicked.connect(self.onislem)
        self.pushButton_7.clicked.connect(self.algoritmasec)
        self.tabWidget.setCurrentIndex(0)#her zaman ilk sayfada acilmasi icin
        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.comboBox_2.addItem("HSV")
        self.comboBox_2.addItem("CIE")
        self.comboBox.addItem("VGG16")
        self.comboBox.addItem("InceptionV3")
        self.comboBox.addItem("ResNet50")
        self.comboBox_3.addItem("RGB")
        self.comboBox_3.addItem("HSV")
        self.comboBox_3.addItem("CIE")
        self.comboBox_3.addItem("RGB ATTIRILMIS")
        self.comboBox_3.addItem("HSV ATTIRILMIS")
        self.comboBox_3.addItem("CIE ATTIRILMIS")
        self.comboBox_4.addItem("0.1")
        self.comboBox_4.addItem("0.2")
        self.comboBox_4.addItem("0.3")
        self.comboBox_5.addItem("2")
        self.comboBox_5.addItem("5")
        self.comboBox_5.addItem("10")
        self.comboBox_6.addItem("1")
        self.comboBox_6.addItem("2")
        self.comboBox_6.addItem("3")
        self.comboBox_6.addItem("4")
        self.comboBox_6.addItem("5")
        self.comboBox_6.addItem("6")
        self.comboBox_6.addItem("7")
        self.comboBox_6.addItem("8")
        self.comboBox_6.addItem("9")
        self.comboBox_6.addItem("10")
        self.comboBox_7.addItem("RGB")
        self.comboBox_7.addItem("HSV")
        self.comboBox_7.addItem("CIE")
        self.comboBox_7.addItem("RGB ATTIRILMIS")
        self.comboBox_7.addItem("HSV ARTTIRILMIS")
        self.comboBox_7.addItem("CIE ARTTIRILMIS")
        self.textEdit_2.setEnabled(False)
        self.textEdit_9.setEnabled(False)
        self.label_10.setText(" ")
        self.pushButton_4.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.comboBox_6.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.pushButton_10.clicked.connect(self.veriarttir)
        
    def importet(self):        
        self.pushButton_4.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.pushButton_10.setEnabled(True)
        file = str(QFileDialog.getExistingDirectory(self,
                                                    "Open a folder",
                                                    expanduser("~"),
                                                    QFileDialog.ShowDirsOnly))
# dosya yolundan dosyanın adını ve uzantısını ayrı ayrı çekmek için path.split() 
        a , b = os.path.split(file)
        self.anaklasor="./"+b+"/"
        #self.anaklasor=file.replace("C:/Users/aylin/OneDrive/Masaüstü/goruntuodev/","./")+"/"
        self.textEdit_2.setText(self.anaklasor)
        self.textEdit_9.setText(self.anaklasor)
        self.directories=os.listdir(self.anaklasor)
        
    def renkuzaylari(self):
        self.label_10.setText(" ")
        for self.directory in self.directories:   
            if(self.radioButton_8.isChecked()):
               self.files=os.listdir(self.anaklasor+self.directory+"/")#klasördeki resimler
               hsvdir="./hsv/"+self.directory+"/"#oluşturulacak resimlerin ekleneceği klasör
               ciedir="./cie/"+self.directory+"/"
               if(self.comboBox_2.currentIndex()==0):
                 if os.path.exists(hsvdir):
                     shutil.rmtree(hsvdir)
                 os.mkdir(hsvdir)    
                 for file_name in self.files:
                     img = cv2.imread(self.anaklasor+self.directory+"/"+file_name,3)
                     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#hsv çevirme
                     cv2.imwrite(hsvdir+file_name,hsv)   
                 
                 
               if(self.comboBox_2.currentIndex()==1):
                 if os.path.exists(ciedir):
                     shutil.rmtree(ciedir)
                 os.mkdir(ciedir) 
                 for file_name in self.files:
                     img = cv2.imread(self.anaklasor+self.directory+"/"+file_name,3)
                     cie=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)#cie çevirme
                     cv2.imwrite(ciedir+file_name,cie) 
                  
                 
            if(self.radioButton_9.isChecked()):   
               arttirildi="./veri/"
               self.files=os.listdir(arttirildi+self.directory+"/")
               hsvArttidir="./hsvArtti/"+self.directory+"/"#oluşturulacak resimlerin ekleneceği klasör
               cieArttidir="./cieArtti/"+self.directory+"/"
               if(self.comboBox_2.currentIndex()==0):
                 if os.path.exists(hsvArttidir):
                     shutil.rmtree(hsvArttidir)
                 os.mkdir(hsvArttidir)    
                 for file_name in self.files:
                     img = cv2.imread(arttirildi+self.directory+"/"+file_name,3)
                     hsvA = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#hsv çevirme
                     cv2.imwrite(hsvArttidir+file_name,hsvA)   
                 
                 
               if(self.comboBox_2.currentIndex()==1):
                 if os.path.exists(cieArttidir):
                     shutil.rmtree(cieArttidir)
                 os.mkdir(cieArttidir) 
                 for file_name in self.files:
                     img = cv2.imread(arttirildi+self.directory+"/"+file_name,3)
                     cieA=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)#cie çevirme
                     cv2.imwrite(cieArttidir+file_name,cieA) 
                    
        if(self.radioButton_9.isChecked()==False and self.radioButton_8.isChecked()==False):
            self.hata = "Lütfen Dönüşüm Yapılacak Bir Veri Seti Seçimi Yapınız!"
            self.error()
        if(self.radioButton_9.isChecked()==True or self.radioButton_8.isChecked()==True):   
            self.basarili = "Renk Dönüşüm İşlemi Başarılı."
            self.success()       
    
    def onislem(self): 
        self.comboBox_6.setEnabled(False)
        self.pushButton_2.setEnabled(True)
        if(self.comboBox_7.currentIndex()==0):#RGB
            self.path=self.anaklasor  
            if(self.radioButton.isChecked()):#RGB -HOLDOUT
                self.holdout()
            if(self.radioButton_2.isChecked()):#RGB-KFOLD
                self.kfold()
        if(self.comboBox_7.currentIndex()==1):#HSV 
            self.path="./hsv"
            if(self.radioButton.isChecked()):#HSV-HOLDOUT
                self.holdout()
            if(self.radioButton_2.isChecked()):#HSV-KFOLD
                self.kfold()  
        if(self.comboBox_7.currentIndex()==2):#CIE
            self.path="./cie"
            if(self.radioButton.isChecked()):#CIE-HOLDUOUT
                self.holdout()
            if(self.radioButton_2.isChecked()):#CIE-KFOLD
                self.kfold()
        if(self.comboBox_7.currentIndex()==3):#RGB DATA AGU
            self.path="./veri"
            if(self.radioButton.isChecked()):#RGB DATA AGU-HOLDOUT 
                self.holdout()
            if(self.radioButton_2.isChecked()):#RGB DATA AGU-KFOLD
                self.kfold()
        if(self.comboBox_7.currentIndex()==4):#HSV DATA AGU
            self.path="./hsvArtti"
            if(self.radioButton.isChecked()):#HSV DATA AGU-HOLDOUT 
                self.holdout()
            if(self.radioButton_2.isChecked()):#HSV DATA AGU-KFOLD
                self.kfold()    
        if(self.comboBox_7.currentIndex()==5):#CIE DATA AGU
            self.path="./cieArtti"
            if(self.radioButton.isChecked()):#CIE DATA AGU-HOLDOUT 
                self.holdout()
            if(self.radioButton_2.isChecked()):#CIE DATA AGU-KFOLD
                self.kfold()        
        if(self.radioButton.isChecked()==False and self.radioButton_2.isChecked()==False):
            self.hata = "Lütfen bir seçim yapınız!"
            self.error()
            
    def holdout(self):
#klasörisimlerini aldım        
        self.cicekler=[]
        self.klasorismi=self.path
        self.directories=os.listdir(self.klasorismi)
        for self.directory in self.directories:
            self.files=os.listdir(self.path+"/"+self.directory+"/")
            self.cicekler.append(self.directory)     
        test_size=self.comboBox_4.currentText()
        y = []
        self.X = []
        cicek_index = -1#sifirdan başlatmak için
        for cicek in self.cicekler: 
            path = os.path.join(self.path, cicek)
            cicek_index+=1
            print("path",path)
            for img in os.listdir(path):               
                resimdizisi = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
                resized_array = cv2.resize(resimdizisi, (224,224))
                self.X.append(resized_array)
                y.append(cicek_index)    
        self.X=np.array(self.X)
        y=np.array(y)          
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=float(test_size), random_state=42)       
        a=self.X_train.shape[1]
        n=self.X_train.shape[2]
        f=self.X_train.shape[3]
        c=self.X_test.shape[1]
        m=self.X_test.shape[2]
        k=self.X_test.shape[3]
        self.X_train= self.X_train.reshape(self.X_train.shape[0], a*n*f)#4 BOYUTU 2 BOYUTA DÜŞÜRMEK İÇİN
        self.X_test = self.X_test.reshape(self.X_test.shape[0], c*m*k)
        self.textEdit_3.setText(str(self.X_train))
        self.textEdit_5.setText(str(self.y_train))
        self.textEdit_4.setText(str(self.X_test))
        self.textEdit_6.setText(str(self.y_test))

    def kfold(self):
        self.comboBox_6.setEnabled(True)
        self.cicekler=[]
        self.klasorismi=self.path
        self.directories=os.listdir(self.klasorismi)
        for self.directory in self.directories:
            self.files=os.listdir(self.path+"/"+self.directory+"/")
            self.cicekler.append(self.directory)    
        y = []
        self.X = []
        cicek_index = -1#sifirdan başlatmak için
        for cicek in self.cicekler: 
            path = os.path.join(self.path, cicek)
            cicek_index+=1
            for img in os.listdir(path):               
                resimdizisi = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
                resized_array = cv2.resize(resimdizisi, (224,224)) 
                self.X.append(resized_array)
                y.append(cicek_index)               
        self.X=np.array(self.X)
        y=np.array(y)   
        n_splits=self.comboBox_5.currentText()
        self.indexal=self.comboBox_6.currentText()
        self.sayac=0
        kf = KFold(n_splits=int(n_splits), random_state=1, shuffle=True)
        for self.train_index , self.test_index in kf.split(self.X):
            self.sayac+=1
            if(self.sayac==int(self.indexal)):
                self.X_train , self.X_test = self.X[self.train_index],self.X[self.test_index]
                self.y_train , self.y_test = y[self.train_index] , y[self.test_index]
                a=self.X_train.shape[1]
                n=self.X_train.shape[2]
                f=self.X_train.shape[3]
                c=self.X_test.shape[1]
                m=self.X_test.shape[2]
                k=self.X_test.shape[3]
                self.X_train= self.X_train.reshape(self.X_train.shape[0], a*n*f)#4 BOYUTU 2 BOYUTA DÜŞÜRMEK İÇİN
                self.X_test = self.X_test.reshape(self.X_test.shape[0], c*m*k)  
                self.textEdit_3.setText(str(self.X_train))
                self.textEdit_5.setText(str(self.y_train))
                self.textEdit_4.setText(str(self.X_test))
                self.textEdit_6.setText(str(self.y_test))
                
    def veriarttir(self):
         path="./flower/"
         directories=os.listdir(path) 
         for i,directory in enumerate(directories):
             print (directory)
             files=os.listdir(path+directory)
             print (files)
             for file_name in files:
                 img = cv2.imread(path+directory+"/"+file_name)
                 if i==0:
                     klasoradi=str("daisy/")
                 if i==1:
                     klasoradi=str("dandelion/")
                 if i==2:
                     klasoradi=str("rose/")
                 if i==3:
                     klasoradi=str("sunflower/")
                 if i==4:
                     klasoradi=str("tulip/")
                 flipp=cv2.flip(img,1)
                 flipp0=cv2.flip(img,0)
                 cv2.imwrite("./veri/"+klasoradi+file_name+"flipped0"+".jpg",flipp0)
                 cv2.imwrite("./veri/"+klasoradi+file_name+"flipped2"+".jpg",img)
                 cv2.imwrite("./veri/"+klasoradi+file_name+"flipped"+".jpg",flipp)
                 print("./veri/"+klasoradi+file_name+"flipped"+".jpg")
         self.basarili = "Veri Arttırma İşlemi Başarılı."
         self.success()
                
    def ikincisayfayagec(self):
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget.setTabEnabled(0,False)
    
    def birincisayfayadon(self):
        self.tabWidget.setTabEnabled(0,True)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabEnabled(1,False)
    
    def algoritmasec(self):
        self.textEdit_7.setText("")
        self.textEdit_8.setText("")
        if(self.radioButton_3.isChecked()):
            self.logisticRegression()
        if(self.radioButton_4.isChecked()):
            self.knn()
        if(self.radioButton_5.isChecked()):
            self.svc()
        if(self.radioButton_6.isChecked()):
            self.nb()
        if(self.radioButton_7.isChecked()):
            self.classifier()
        if(self.radioButton_3.isChecked()==False and self.radioButton_4.isChecked()==False
           and self.radioButton_5.isChecked()==False and self.radioButton_6.isChecked()==False
           and self.radioButton_7.isChecked()==False):
            self.hata = "Lütfen bir seçim yapınız!"
            self.error()    
        self.matrix()
        self.roc()
    
    def logisticRegression(self):   
        self.model = LogisticRegression()
        self.model.fit(self.X_train,self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        acc = self.model.score(self.X_test,self.y_test)*100
        print("Logistic Regression Accuracy {:.2f}%".format(acc))  
        print("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
        self.textEdit_7.setText("{:.2f}%".format(acc))
        self.textEdit_8.setText("Gerçek Veriler:\n"+str(self.y_test)+
                                "\nTahmin Edilen Veriler:\n"+str(self.y_pred))
        
    def knn(self):
        self.model = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
        self.model.fit(self.X_train,self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        acc = self.model.score(self.X_test,self.y_test)*100
        print("KNN Accuracy {:.2f}%".format(acc))
        print("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))    
        self.textEdit_7.setText("{:.2f}%".format(acc))
        self.textEdit_8.setText("Gerçek Veriler:\n"+str(self.y_test)+
                                "\nTahmin Edilen Veriler:\n"+str(self.y_pred))
    def svc(self):
        self.model=svc(probability=True)#probability=True roc eğrisi için
        self.model.fit(self.X_train,self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        acc = self.model.score(self.X_test,self.y_test)*100
        print("SVC Accuracy {:.2f}%".format(acc)) 
        print("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
        self.textEdit_7.setText("{:.2f}%".format(acc))
        self.textEdit_8.setText("Gerçek Veriler:\n"+str(self.y_test)+
                                "\nTahmin Edilen Veriler:\n"+str(self.y_pred))
    def nb(self):
         self.model= GaussianNB()
         self.model.fit(self.X_train,self.y_train)
         self.y_pred = self.model.predict(self.X_test)
         acc = self.model.score(self.X_test,self.y_test)*100
         print("NB Accuracy {:.2f}%".format(acc))
         print("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
         self.textEdit_7.setText("{:.2f}%".format(acc))
         self.textEdit_8.setText("Gerçek Veriler:\n"+str(self.y_test)+
                                "\nTahmin Edilen Veriler:\n"+str(self.y_pred))
    def classifier(self):
        self.model= DecisionTreeClassifier(criterion = 'entropy')
        self.model.fit(self.X_train,self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        acc = self.model.score(self.X_test,self.y_test)*100
        print("DecisionTreeClassifier Accuracy {:.2f}%".format(acc))
        print("Gerçek Veriler:\n"+str(self.y_test)+"\nTahmin Edilen Veriler:\n"+str(self.y_pred))
        self.textEdit_7.setText("{:.2f}%".format(acc))
        self.textEdit_8.setText("Gerçek Veriler:\n"+str(self.y_test)+
                                "\nTahmin Edilen Veriler:\n"+str(self.y_pred))
    def matrix(self):
        plt.figure(figsize=(4,4))
        plt.subplot(1, 2, 1)
        splt.plot_confusion_matrix(self.y_test, self.y_pred)
        plt.savefig("./MOA_matrix.png")
        self.pixmap = QPixmap("./MOA_matrix.png")
        self.label_19.setPixmap(self.pixmap)
   
    def roc(self):
        self.pred_prob = self.model.predict_proba(self.X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresh = dict()
        n_classes=5
        for i in range(n_classes):    
            fpr[i], tpr[i], thresh[i] = roc_curve(self.y_test, self.pred_prob[:,i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])  
#        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#        mean_tpr = np.zeros_like(all_fpr)
#        for i in range(n_classes):
#            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#        mean_tpr /= n_classes
#        fpr["macro"] = all_fpr
#        tpr["macro"] = mean_tpr
#        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw=2
        plt.figure(figsize=(7,4.8))
#        plt.plot(fpr["macro"], tpr["macro"],
#                 label='macro-average ROC curve (area = {0:0.2f})'
#                       ''.format(roc_auc["macro"]),
#                 color='green', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','yellow'])
        for i, color in zip(range(n_classes), colors):
            print(i)
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.annotate('Random Guess',(.5,.48),color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC EĞRİSİ')
        plt.legend(loc="lower right")
        plt.savefig("./MOA_roc.png")
        self.pixmap = QPixmap("./MOA_roc.png")
        self.label_20.setPixmap(self.pixmap)
        plt.show()
    
    def roctransfer(self):
        self.pred = self.model.predict_generator(self.Test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresh = dict()
        n_classes=5
        for i in range(n_classes):    
            fpr[i], tpr[i], thresh[i] = roc_curve(self.Test.classes, self.pred[:,i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])  
#        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#        mean_tpr = np.zeros_like(all_fpr)
#        for i in range(n_classes):
#            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#        mean_tpr /= n_classes
#        fpr["macro"] = all_fpr
#        tpr["macro"] = mean_tpr
#        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw=2
        plt.figure(figsize=(7,4.8))
#        plt.plot(fpr["macro"], tpr["macro"],
#                 label='macro-average ROC curve (area = {0:0.2f})'
#                       ''.format(roc_auc["macro"]),
#                 color='green', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','yellow'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.annotate('Random Guess',(.5,.48),color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC EĞRİSİ')
        plt.legend(loc="lower right")
        plt.savefig("./tl_roc.png")
        self.pixmap = QPixmap("./tl_roc.png")
        self.label_7.setPixmap(self.pixmap)
        plt.show()    
        
    def error(self):
        msg = QMessageBox()
        msg.setWindowTitle("Uyarı")
        msg.setText(self.hata)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()
    
    def success(self):
        msg = QMessageBox()
        msg.setWindowTitle("Başarılı")
        msg.setText(self.basarili)
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_() 
    
    def ucuncusayfayagec(self):
        self.tabWidget.setTabEnabled(2,True)
        self.tabWidget.setCurrentIndex(2)
        self.tabWidget.setTabEnabled(1,False)
   
    def ikincisayfayadon(self):
        self.tabWidget.setTabEnabled(1,True)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget.setTabEnabled(2,False)
   
    def uygula(self):
        if(self.comboBox_3.currentIndex()==0): 
            self.klasor=self.anaklasor
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model()    
        if (self.comboBox_3.currentIndex()==1):
            self.klasor="./hsv"
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()                
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model()  
        if(self.comboBox_3.currentIndex()==2):
            self.klasor="./cie"
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model() 
        if(self.comboBox_3.currentIndex()==3):
            self.klasor="./veri"
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model()        
        if(self.comboBox_3.currentIndex()==4):
            self.klasor="./hsvArtti"
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model()
        if(self.comboBox_3.currentIndex()==5):
            self.klasor="./cieArtti"
            if(self.comboBox.currentIndex()==0):
                self.vgg16model()
            if(self.comboBox.currentIndex()==1):
                self.InceptionV3model() 
            if(self.comboBox.currentIndex()==2):
                self.resnet50model()
    
    def vgg16model(self):    
        self.textEdit.setText(" ")
        IMAGE_SIZE = [224, 224]
        vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in vgg.layers:
          layer.trainable = False
        folders = glob('./flower/*')
        x = Flatten()(vgg.output)
        prediction = Dense(len(folders), activation='softmax')(x)
        self.model = Model(inputs=vgg.input, outputs=prediction) 
        self.model.summary()

#modelin derlenmesi       
        self.model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        )
        data_generator = ImageDataGenerator(
                                                rescale = 1. / 255, 
                                                validation_split = 0.2)#hold-out yöntemiyle veri setini ayırır
                
        self.Train=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=True,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="training")    
                             
        self.Test=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=False,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="validation")
            
        
# fit model
        self.model_fit = self.model.fit_generator(
          self.Train,
          validation_data=self.Test,
          epochs=10,
          steps_per_epoch=len(self.Train),
          validation_steps=len(self.Test)
        )    
        scores = self.model.evaluate(self.Test, verbose=0)#verbose=ayrıntılı
        print("Accuracy: %.2f%%" % (scores[1]*100))#bunu textEdita yazdır
        self.textEdit.setText("%.2f%%" % (scores[1]*100))
        
        self.X_train, self.y_train = next(self.Train)
        self.X_test, self.y_test = next(self.Test)            
        self.tahmin=self.model.predict_generator(self.Train)
        print(self.tahmin)

#başarı grafiği
        self.basarigrafik()

#loss grafiği     
        self.losgrafik()

#h5 dosyası        
        self.model.save('flowersVGG16_model.h5')

#Confusion matrix
        self.plot_confusion_matrix()
#roc eğrisi        
        self.roctransfer()
        
    def basarigrafik(self):
        plt.clf()#temizlemek için
        plt.figure(figsize=(12,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(self.model_fit.history['acc'])
        plt.plot(self.model_fit.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./AccVal_acc.png')
        self.pixmap = QPixmap("./AccVal_acc.png") 
        self.label_5.setPixmap(self.pixmap)
    
    def losgrafik(self):
        plt.clf()#temizlemek için
        plt.figure(figsize=(12,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(self.model_fit.history['loss'])
        plt.plot(self.model_fit.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./lossVal.png')
        self.pixmap = QPixmap("./lossVal.png") 
        self.label_4.setPixmap(self.pixmap)
        
    def plot_confusion_matrix(self):
        classes = []
        for i in self.Train.class_indices:
             classes.append(i)
        tahmin = self.model.predict_generator(self.Test)
        y_pred = np.argmax(tahmin, axis=1)
        print('Confusion Matrix')
        cm = confusion_matrix(self.Test.classes, y_pred)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        print("thresh",thresh)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('./confusionmatrix.png')
        self.pixmap = QPixmap("./confusionmatrix.png") 
        self.label_6.setPixmap(self.pixmap)
        
    def InceptionV3model(self):
        self.textEdit.setText(" ")
        IMAGE_SIZE = [224, 224]
        increptionv3 = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in  increptionv3.layers:
          layer.trainable = False
        folders = glob('./flower/*')
        x = Flatten()( increptionv3.output)#BURDA HATA
        prediction = Dense(len(folders), activation='softmax')(x)
        self.model = Model(inputs= increptionv3.input, outputs=prediction) 
        self.model.summary()
        
#modelin derlenmesi       
        self.model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        )
        data_generator = ImageDataGenerator(
                                                rescale = 1. / 255, 
                                                validation_split = 0.2)#hold-out yöntemiyle veri setini ayırır
                
        self.Train=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=True,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="training")    
                             
        self.Test=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=False,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="validation")
       
# fit model
        self.model_fit = self.model.fit_generator(
          self.Train,
          validation_data=self.Test,
          epochs=20,
          steps_per_epoch=len(self.Train),
          validation_steps=len(self.Test)
        )    
        scores = self.model.evaluate(self.Test, verbose=0)#verbose=ayrıntılı
        print("Accuracy: %.2f%%" % (scores[1]*100))#bunu textEdita yazdır
        self.textEdit.setText("%.2f%%" % (scores[1]*100))
        
        X_train, y_train = next(self.Train)
        X_test, y_test = next(self.Test) 
        tahmin=self.model.predict_generator(self.Train)
        print(tahmin)
        
#başarı grafiği
        self.basarigrafik()
        
#loss grafiği     
        self.losgrafik()
        
#h5 dosyası        
        self.model.save('flowersInception_model.h5')
        
#Confusion matrix
        self.plot_confusion_matrix()
#roc eğrisi        
        self.roctransfer()
    def resnet50model(self):    
        self.textEdit.setText(" ")
        IMAGE_SIZE = [224, 224]
        resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in resnet.layers:
          layer.trainable = False   
        folders = glob('./flower/*')
        x = Flatten()(resnet.output)
        prediction = Dense(len(folders), activation='softmax')(x)
        self.model = Model(inputs=resnet.input, outputs=prediction)
        self.model.summary()

#Modelin derlenmesi
        self.model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        )
        data_generator = ImageDataGenerator(
                                                rescale = 1. / 255, 
                                                validation_split = 0.2)#hold-out yöntemiyle veri setini ayırır
                
        self.Train=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=True,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="training")    
        self.Test=data_generator.flow_from_directory(self.klasor,
                                              target_size=(224,224),
                                              shuffle=False,
                                              batch_size=20,
                                              class_mode='categorical',
                                              subset="validation")

# fit model
        self.model_fit = self.model.fit_generator(
          self.Train,
          validation_data=self.Test,
          epochs=20,
          steps_per_epoch=len(self.Train),
          validation_steps=len(self.Test)
        )    
        scores = self.model.evaluate(self.Test, verbose=0)#verbose=ayrıntılı
        print("Accuracy: %.2f%%" % (scores[1]*100))#bunu textEdita yazdır
        self.textEdit.setText("%.2f%%" % (scores[1]*100))

#başarı grafiği
        self.basarigrafik()

#loss grafiği     
        self.losgrafik()

#h5 dosyası        
        self.model.save('flowersResNet50_model.h5')
        
#Confusion matrix
        self.plot_confusion_matrix()    
#roc eğrisi
        self.roctransfer()        