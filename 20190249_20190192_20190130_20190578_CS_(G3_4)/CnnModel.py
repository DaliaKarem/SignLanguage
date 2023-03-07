
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")

class CnnModel:
    
    def expirment1(self,filtersC,unitsC,inputLayer,pooling2):
        #To convert to 1 dimentail Aarray
            flattenLayer = tf.keras.layers.Flatten()(pooling2)
            denseLayer1 = tf.keras.layers.Dense(units=unitsC, activation='relu')(flattenLayer)
            denseLayer2 = tf.keras.layers.Dense(units=unitsC, activation='relu')(denseLayer1)
            outputLayer = tf.keras.layers.Dense(units=10, activation='softmax')(denseLayer2)
            return  tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
        
    def expiremnt2(self,filtersC,unitsC,inputLayer,pooling2):
            convolution3 = tf.keras.layers.Conv2D(filters=filtersC, kernel_size=(5, 5), activation='relu')(pooling2)
            pooling3 = tf.keras.layers.AveragePooling2D()(convolution3)
            flattenLayer = tf.keras.layers.Flatten()(pooling3)
            denseLayer1 = tf.keras.layers.Dense(units=unitsC, activation='relu')(flattenLayer)
            denseLayer2 = tf.keras.layers.Dense(units=unitsC, activation='relu')(denseLayer1)
            denseLayer3 = tf.keras.layers.Dense(units=unitsC, activation='relu')(denseLayer2)
            outputLayer = tf.keras.layers.Dense(units=10, activation='softmax')(denseLayer3)
            return tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
        
    def cnnModel(self,colorChannel,filtersC,unitsC,exprementC):
        """
        First Make Input Layer -->one input image is 64*64
        Make First Layer
        Step 1-->Convential and RELU
        Where is the feature in the image by multiplty feature with the image
        and then make feature map,and make RELU to convert the all negative values in feature map to zero
        #There Exist Two methods to make Pooling 1-MAX 2-AVERAGE
        #Using Average-->reduce the dimensions of the feature maps.
        """
        inputLayer=tf.keras.Input(shape=(64, 64,colorChannel))
        convolution1 = tf.keras.layers.Conv2D(filters=filtersC, kernel_size=(5, 5), activation='relu')(inputLayer)
        pooling1 = tf.keras.layers.AveragePooling2D()(convolution1)
        convolution2 = tf.keras.layers.Conv2D(filters=filtersC, kernel_size=(5, 5), activation='relu')(pooling1)
        pooling2 = tf.keras.layers.AveragePooling2D()(convolution2)
        if(exprementC==2):
            return self.expiremnt2(filtersC,unitsC,inputLayer,pooling2)
        else:
            return self.expirment1(filtersC,unitsC,inputLayer,pooling2)

    def fitModel(self,model,xTrain,yTrain,verC):
        model.compile(
            optimizer='adam',# Optimizer that implements the Adadelta algorithm.(when compile)
            loss='sparse_categorical_crossentropy',#Loss function
            metrics=['accuracy']# judge the performance of your model.
        )  
        fittingModel = model.fit(
            xTrain,
            yTrain,
            #Make Valdation Part From the Trainng Data 
            batch_size=100,
            epochs=20,
            verbose=verC
        )

    def predict(self,cnnModel,xTrain,xTest,yTarin,yTest):
        
        self.fitModel(cnnModel,xTrain,yTarin,0)
        print("\n\n\n--------------------CNN Model Summary Of Best Expirment -----------------------------\n\n")
        print(cnnModel.summary(),"\n\n")
        y_pred =cnnModel.predict(xTest,verbose=0)
        y_pred_classes = [np.argmax(element) for element in y_pred]
        print("Classification Report: \n", classification_report(yTest, y_pred_classes))
        return accuracy_score(yTest,y_pred_classes)
    
    def makeExpairment(self,xTrain,xTest,yTarin,yTest,colorChannel):  
        
        #--------------------------Expirment 1--------------------------------   
        archFolds=self.startLearn(xTrain,yTarin,colorChannel,20,120,1)
        #make a average accuarcy
        averageExpirment1=np.average(archFolds)
        print("\n\nThe Average of CNN Expirment 1 : ",averageExpirment1)
        
        #---------------------------------Expirment 2-----------------------------
        archFolds=self.startLearn(xTrain,yTarin,colorChannel,30,130,2)
        #make a average accuarcy
        averageExpirment2=np.average(archFolds)
        print("\n\nThe Average  Accurcay ANN of Expirment 2 : ",averageExpirment2)
        
        if(averageExpirment1>averageExpirment2):
            #Use that as Best Archictiure In CNN (Exirnment 1)
            model=self.cnnModel(colorChannel,20,120,1)
            print("\n\nThe Result Of Best CNN Model To Predict Is: ")
            return self.predict(model,xTrain,xTest,yTarin,yTest)
        else:
            model=self.cnnModel(colorChannel,colorChannel,30,130,2)
            print("\n\nThe Result Of Best CNN Alicture To Predict Is: ")
            #Use That as Best Archicture In CNN (Exirnment 2)  
            return self.predict(model,xTrain,xTest,yTarin,yTest)
        
    def startLearn(self,xTrain,yTarin,colorChannel,filtersC,unitsC,exprementC):
        archFolds=[]
        print("----------------------CNN Expiremnts---------------------------------")
        kf = KFold(n_splits=5)
        for currentK, (train_index, test_index) in enumerate(kf.split(xTrain, yTarin)):
            cnnModel=self.cnnModel(colorChannel,filtersC,unitsC,exprementC)
            print("\nTrainng In Fold: ",currentK+1,"\n")
            self.fitModel(cnnModel,xTrain[train_index, :],yTarin[train_index],2)
            xVal=xTrain[test_index, :]
            yVal=yTarin[test_index]
            print("\nValidation (Accuracy) In Fold : ",currentK+1," ",cnnModel.evaluate(xVal,yVal)[0])
            print("\n====================================\n")
            archFolds.append(cnnModel.evaluate(xVal,yVal)[0])
        return archFolds
            
    