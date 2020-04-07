import numpy as np
import pandas as pd
df=pd.read_csv("studentdata.csv",sep=';')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC #support vector machine Classifier Model
""" split data into Training and Testing Sets """
def split_data(X,Y):
    return train_test_split(X,Y,test_size=0.2,random_state=0)
""" Confusion Matrix """
def confuse(y_true,y_pred):
    cm=confusion_matrix(y_true=y_true,y_pred=y_pred)
    #print("\nConfusion Matrix: \n",cm)
    fpr(cm)
    ffr(cm)
""" False Pass Rate """
def fpr(confusion_matrix):
    fp=confusion_matrix[0][1]
    tf=confusion_matrix[0][0]
    rate=float(fp)/(fp+tf)
    print("False pass rate :",rate)
""" False fail rate """
def ffr(confusion_matrix):
    ff=confusion_matrix[1][0]
    tp=confusion_matrix[1][1]
    rate=float(ff)/(ff+tp)
    print("False Fail Rate :",rate)
    return rate
""" Train Model and Print score """
def train_and_score(X,y):
    X_train,X_test,y_train,y_test=split_data(X,y)
    clf=Pipeline([
            ('reduce dim', SelectKBest(chi2,k=2)),
            ('train',LinearSVC(C=100))])
    scores=cross_val_score(clf,X_train,y_train,cv=5,n_jobs=2)
    print("Mean Model Accuracy :",np.array(scores).mean())
    clf.fit(X_train,y_train)
    confuse(y_test,clf.predict(X_test))
    print()
    
    """ Main Program """
def main():
    print("\nStudent Performance Prediction")
        #for each feature, encode to categorical values
    class_le=LabelEncoder()
    for column in df:
        df[column]=class_le.fit_transform(df[column].values)
    for i,column in df.itercolumnss():
        if column[ETP_100]<=100 and column[ETP_100]>=50:
            print("Great")
            