from DataSet import *
from DnnModel import *
from CnnModel import *
from SVM import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    dataSet=DataSet('Dataset')
    #DNN MODEL
    dnnModel=DnnModel()
    #1-->Noriamize (not bonus part)
    print("---------------------DNN Model--------------------\n")
    xTrain,xTest,yTarin,yTest=dataSet.mainProcess(1)    
    dnnAccuracy=dnnModel.makeExpairment(xTrain,xTest,yTarin,yTest)
    print("\n\n\n\n\n\n")
    
    print("---------------------CNN Model--------------------\n")
    #CNN MODEL 
    #2-->Noriamize (using bonus part)
    #3-->RGB (3Colors)
    cnnModel=CnnModel()
    xTrain,xTest,yTarin,yTest=dataSet.mainProcess(2)
    cnnAccuracy=cnnModel.makeExpairment(xTrain,xTest,yTarin,yTest,3)
    
    print("---------------------SVM--------------------\n")
    
    #SVM Accuracy
    xTrain,xTest,yTarin,yTest=dataSet.mainProcess(1)  
    SVM=SVM_Model()
    svmAccuracy=SVM.Compare(xTrain,yTarin,xTest,yTest)
    
    print("The Accuarcy Of Three Models: \n")
    print("The Accuarcy Of CNN: ",cnnAccuracy)
    print("\nThe Accuarcy Of DNN: ",dnnAccuracy)
    print("\nThe Accuarcy Of SVM: ",svmAccuracy)
    
    

