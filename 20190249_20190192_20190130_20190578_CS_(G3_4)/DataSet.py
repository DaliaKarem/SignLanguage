import io
import sys

from os import listdir

import PIL
import cv2
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings("ignore")


class DataSet:

    def __init__(self,dataSetPath):
        self.dataSetPath=dataSetPath
    def normailzeStep(self,imagePath):
        #Read Image Using CV2
        originalImage = cv2.imread(imagePath)
        grayImage= cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        #Resize the image to 64*64#
        #Resize Because exist image not 100*100
        resizedImage = cv2.resize(grayImage,(64,64), interpolation = cv2.INTER_AREA)
        return (resizedImage/255)

    def getData(self,id):
        #[0,1,2,3,4,5,6,7,8,9]
        folderDigitNames = listdir(self.dataSetPath) 
        X = [] #ThE TOTAL IMAGES IN DATASET
        Y = [] #TOTAL OUTPUT From [0--9]
        for folderName, oneFolder in enumerate(folderDigitNames):
            oneFolderPath = self.dataSetPath+'/'+oneFolder
            for imageName in listdir(oneFolderPath):
                image=0
                if(id==1):
                    image = self.normailzeStep(oneFolderPath+'/'+imageName)
                else:
                    image=cv2.imread(oneFolderPath+'/'+imageName)
                    image = cv2.resize(image,(64,64), interpolation = cv2.INTER_AREA)
                X.append(image)
                Y.append(folderName)

        #X[0]-->that is the one image.-->that the label is the Y[0]
        X=np.array(X)
        Y=np.array(Y)

        return X,Y
    
    def divideDataSet(self,X,Y,trainSize):
        xTrain,xTest,yTarin,yTest=train_test_split(X,Y,train_size=trainSize)
        return xTrain,xTest,yTarin,yTest
        
    def avg(self,imagePath):
        global red,green,blue,count
        originalImage =Image.open(imagePath)
        resizedImage = originalImage.resize((64,64))
        #for one image
        for r in range(64):
            for c in range(64):
                pix=resizedImage.load()
                red=red+pix[r,c][0]
                green=green+pix[r,c][1]
                blue=blue+pix[r,c][2]
        count+=(64*64)
        return resizedImage
    def sub(self,image):
        #for one image
        pix=image.load()
        global aerageRed,averageBlue,averageGreen
        for r in range(64):
            for c in range(64):
                change1=pix[r,c][0]-aerageRed
                change2=pix[r,c][1]-averageGreen
                change3=pix[r,c][2]-averageBlue
                pix[r,c]=(int(change1),int(change2),int(change3))
        return image
    
    def averageNormaliztion(self,X_data):
        #[0,1,2,3,4,5,6,7,8,9]
        avg1=np.mean(X_data,axis=0)
        subtract1=X_data-avg1
        normValue_X=subtract1/255
        return normValue_X
    

    def mainProcess(self,id):

        if(id==1):
            #Part 1
            X,Y=self.getData(1)
            return self.divideDataSet(X,Y,0.8)
        else:
            #Bonus Part
            X,Y=self.getData(2)
            newX=self.averageNormaliztion(X)
            return self.divideDataSet(newX,Y,0.8)
            





    
