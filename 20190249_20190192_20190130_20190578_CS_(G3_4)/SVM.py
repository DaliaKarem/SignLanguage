from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


class SVM_Model:

    
    def Compare(self,X_train,Y_train,X_test,Y_test):
        # normalization
        classifier=SVC(kernel='rbf',random_state=0)
        arr=X_train.reshape(X_train.shape[0],(X_train.shape[1]*X_train.shape[2]))
        classifier.fit(arr,Y_train)
        arr2=X_test.reshape(X_test.shape[0],(X_test.shape[1]*X_test.shape[2]))
        y_pred=classifier.predict(arr2)
        c=confusion_matrix(Y_test,y_pred)
        accuracy=accuracy_score(Y_test,y_pred)
        return accuracy