
# Load libraries
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, explained_variance_score,
                             f1_score, max_error, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def create_model(data, target, bool_classification=False):
    """
    creates, validates and explains model
    """
    # creates training and testdata
    X_train, Y_train, X_validation, Y_validation, columns_X = prepare_model_dataset(
        data=data, target=target, bool_classification=bool_classification)

    if bool_classification:
        models = []
        # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        # models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        # models.append(('SVM', SVC(gamma='auto')))
        models.append(('RF', RandomForestClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))
        # models.append(('RFR',RandomForestRegressor()))

        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            print('-'*100)
            # run classification model

            model = run_model_classifier(model=model, X_train=X_train, Y_train=Y_train, X_validation=X_validation,
                                         Y_validation=Y_validation, results=results, name=name, names=names)
        print('-'*100)

    else:
        models = []
        models.append(('LR', LinearRegression()))
        models.append(('RFR', RandomForestRegressor()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            print('-'*100)

            # run regression model
            model = run_model_regression(model=model, X_train=X_train, Y_train=Y_train, X_validation=X_validation,
                                         Y_validation=Y_validation, results=results, name=name, names=names)
        print('-'*100)

    # Explains Tree Model
    try:
        explain_tree_model(model=model, X_validation=X_validation,
                      columns_X=columns_X)
    except:
        print('Kein Baum-Modell')
        pass
    print('End')


def prepare_model_dataset(data, target, bool_classification=True):
    """
    creates an model dataset
    """
    # drop non-numeric data
    data = data[data.columns[(data.dtypes != 'object')]]
    
    # drop choosen columns
    data = data.drop(['ID', 'OBJECTID', 'SECOND', 'DAY', 'MONTH', 'HOUR', 'MINUTE', 'REGION_CODE',
                     'YEAR_delete', 'COUNTRY_delete', 'LATITUDE', 'LONGITUDE', 'YEAR'], axis=1, errors='ignore')
    
    # handle missing data
    #data = data.fillna(data.mean())
    data = data.fillna(0)

    # save names
    X = data.drop(target, axis=1)
    columns_X = X.columns
    
    # set target value
    if bool_classification:
        y = data[target] > 0
    else:
        y = data[target]

    # divide data in test and train and X und y
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1)

    # print partition
    print('Train: \n', 'True:', Y_train.sum(), '\n',
          'False:', len(Y_train)-Y_train.sum(), '\n')
    print('Test: \n', 'True:', Y_validation.sum(), '\n',
          'False:', len(Y_validation)-Y_validation.sum(), '\n')

    # standard data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    # calibrate data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    return X_train, Y_train, X_validation, Y_validation, columns_X


def explain_tree_model(model, X_validation, columns_X):
    """
    explain tree models with feature importance and shap values 
    """

    fi = pd.Series(model.feature_importances_, index=columns_X)
    
    # calculate feature importances
    print(fi.sort_values(ascending=False))

    # calculate shap values
    rf_shap_values = shap.TreeExplainer(model).shap_values(X_validation)
    X_validation = pd.DataFrame(
        data=X_validation, columns=columns_X)

    shap.summary_plot(rf_shap_values, X_validation)


def run_model_classifier(model, X_train, Y_train, X_validation, Y_validation, results=[], name='model', names=[]):
    """
    runs and validates classification models
    """

    # divide Dataset
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    
    # run kfold model
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    
    # save results
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # fit model
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # validate scores
    N = 4
    print('ACC: ', np.round(accuracy_score(Y_validation, predictions), N))
    print('ROC: ', np.round(roc_auc_score(Y_validation, predictions), N))
    print('F1: ', np.round(f1_score(Y_validation, predictions), N))
    print()
    print(np.round(confusion_matrix(
        Y_validation, predictions, normalize='true'), 3))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    return model


def run_model_regression(model, X_train, Y_train, X_validation, Y_validation, results=[], name='model', names=[]):
    """
    runs and validates regression models
    """

    
    #divide datasets
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=10, scoring='r2')
    
    # save results
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # fit model
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # validate scores
    N = 4
    print('max_error: ', np.round(max_error(Y_validation, predictions), N))
    print('expl_varaiance: ', np.round(
        explained_variance_score(Y_validation, predictions), N))
    print('mean_sqr_log: ', np.round(
        mean_squared_error(Y_validation, predictions), N))
    print('mean_absolute: ', np.round(
        mean_absolute_error(Y_validation, predictions), N))
    print('R2: ', np.round(r2_score(Y_validation, predictions), N))

    return model
