
# Load libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def create_model(data,target):

    data = data[data.columns[(data.dtypes!='object')]]
    data = data.drop(['ID','OBJECTID','SECOND','DAY','MONTH','HOUR','MINUTE','REGION_CODE','YEAR_delete','COUNTRY_delete'],axis=1,errors='ignore')
    #data = data.fillna(data.mean())
    data = data.fillna(0)
    X = data.drop(target,axis=1)

    #TODO Target
    y = data[target] > 0
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1,stratify=y) 
    
    print('Train: \n','True:',Y_train.sum(),'\n','False:',len(Y_train)-Y_train.sum(),'\n')
    print('Test: \n','True:',Y_validation.sum(),'\n','False:',len(Y_validation)-Y_validation.sum(),'\n')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)



    models = []
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('RF',RandomForestClassifier()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        print('-'*100)
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)

        N = 4
        print('ACC: ',np.round(accuracy_score(Y_validation, predictions),N))
        print('ROC: ',np.round(roc_auc_score(Y_validation, predictions),N))
        print('F1: ',np.round(f1_score(Y_validation, predictions),N))
        print()
        print(np.round(confusion_matrix(Y_validation, predictions,normalize='true'),3))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
    print('-'*100)

    fi = pd.Series(model.feature_importances_,index=X.columns)
    print(fi.sort_values(ascending=False))
    print('End')
        
