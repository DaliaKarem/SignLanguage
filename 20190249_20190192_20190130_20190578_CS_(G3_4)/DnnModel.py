
import tensorflow as tf
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
class DnnModel:
    
    def dnnModel(self,modelCount,unitsCount):
        """
        First Layer Is the input layer-->
            that is the features when make that each image pixel is feature
        #Hidden Layer 1 --> that is related to the wights
        #units-->Is the Neures that is number of (Z). 
        #relu-->that is the to get max from each layer. 
        #Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities,
        #Choose units=10-->Becasue we have elemnents from [0--9](10 classes only)
        """
        model = tf.keras.Sequential()
        #One Input Layer-->Make a Flaaten to convert 2d to 1d
        model.add(tf.keras.layers.Flatten(input_shape=(64,64,1))),
        #Two Hidden Layer
        model.add(tf.keras.layers.Dense(units=unitsCount, activation='relu'))
        model.add(tf.keras.layers.Dense(units=unitsCount, activation='relu'))
        
        if(modelCount==2):
            #Addiantial Hidden Layer
            model.add(tf.keras.layers.Dense(units=unitsCount, activation='relu'))
            
        #One Output Layer
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
        
        return model
    def fitModel(self,model,xTrain,yTrain,verC): 
        model.compile(
            optimizer='adam',# Optimizer that implements the Adadelta algorithm.(when compile)
            loss='sparse_categorical_crossentropy',#Loss function
            metrics=['accuracy']# judge the performance of your model.
        )
        model.fit(
            xTrain,
            yTrain,
            #Make Valdation Part From the Trainng Data 
            batch_size=100,
            epochs=70,
            verbose=verC,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    #Stop when the valdation loss after 5 iterations not improve
                    patience=10,
                    restore_best_weights=True #Save the best if condation satiffy.
                )
            ]
        ) 
    
    def startLearn(self,xTrain,yTarin,id,units):
        archFolds=[]
        print("----------------------DNN Expiremnts---------------------------------")
        kf = KFold(n_splits=6)
        for currentK, (train_index, test_index) in enumerate(kf.split(xTrain, yTarin)):
            dnnModel=self.dnnModel(id,units)
            print("\nTrainng In Fold: ",currentK+1,"\n")
            self.fitModel(dnnModel,xTrain[train_index, :],yTarin[train_index],2)
            xVal=xTrain[test_index, :]
            yVal=yTarin[test_index]
            print("\nValidation (Accuracy) In Fold : ",currentK+1," ",dnnModel.evaluate(xVal,yVal)[0])
            print("\n====================================\n")
            archFolds.append(dnnModel.evaluate(xVal,yVal)[0])
        return archFolds
    
    def predict(self,dnnModel,xTrain,xTest,yTarin,yTest):
        
        self.fitModel(dnnModel,xTrain,yTarin,0)
        print("\n\n\n--------------------DNN Model Summary Of Best Expirment -----------------------------\n\n")
        print(dnnModel.summary(),"\n\n")
        y_pred = dnnModel.predict(xTest,verbose=0)
        y_pred_classes = [np.argmax(element) for element in y_pred]
        print("Classification Report: \n", classification_report(yTest, y_pred_classes))
        return accuracy_score(yTest,y_pred_classes)
    
    def makeExpairment(self,xTrain,xTest,yTarin,yTest):  
        
        #--------------------------Expirment 1--------------------------------   
        archFolds=self.startLearn(xTrain,yTarin,1,110)
        #make a average accuarcy
        averageExpirment1=np.average(archFolds)
        print("\n\nThe Average of Expirment 1 : ",averageExpirment1)
        
        #---------------------------------Expirment 2-----------------------------
        archFolds=self.startLearn(xTrain,yTarin,2,150)
        #make a average accuarcy
        averageExpirment2=np.average(archFolds)
        print("\n\nThe Average  Accurcay of Expirment 2 : ",averageExpirment2)
        
        if(averageExpirment1>averageExpirment2):
            #Use that as Best Archictiure In DNN (Exirnment 1)
            model=self.dnnModel(1,110)
            print("\n\nThe Result Of Best DNN Model To Predict Is: ")
            return self.predict(model,xTrain,xTest,yTarin,yTest)
        else:
            model=self.dnnModel(2,150)
            print("\n\nThe Result Of Best Alicture To Predict Is: ")
            #Use That as Best Archicture In DNN (Exirnment 2)  
            return self.predict(model,xTrain,xTest,yTarin,yTest)
        
    