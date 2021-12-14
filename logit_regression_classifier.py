# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:32:25 2021

@author: xiaoyin chang
"""


import csv
import os 
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data with pandas
os.getcwd()
os.chdir('C:/Users/xiaoyin chang/OneDrive - Aalborg Universitet/Dokumenter/Python Scripts/P1dataset')

types_of_encoding = ["utf8", "cp1252"]
for encoding_type in types_of_encoding: #for solving UnicodeDecodeError
    with codecs.open('players_21.csv', 'r', encoding=encoding_type, errors='replace') as csvfile:
        reader = csv.DictReader(csvfile)
        df = pd.DataFrame(reader)
        
#Training a logit classfier for ST position as an example

ST = df[df['player_positions'].str.contains('ST')].loc[:,['pace', 'shooting', 'dribbling', 'age']]
ST['ST_good'] = np.where((pd.to_numeric(ST['pace']) >= 77) | (pd.to_numeric(ST['shooting']) >= 70) | (pd.to_numeric(ST['dribbling']) >= 70) & (pd.to_numeric(ST['age']) <= 29), True, False)
ST_good_per = round(ST['ST_good'].values.sum()/ST.shape[0],3)


CF = df[df['player_positions'].str.contains('CF')].loc[:,['pace', 'shooting', 'dribbling', 'age']]
CF['CF_good'] = np.where((pd.to_numeric(CF['pace']) >= 80 ) | (pd.to_numeric(CF['shooting']) >= 72) & (pd.to_numeric(CF['age']) <= 29) | (pd.to_numeric(CF['dribbling']) >= 72), True, False) 
CF_good_per = round(CF['CF_good'].values.sum()/ CF.shape[0],3)

    
RW = df[df['player_positions'].str.contains('RW')].loc[:, ['pace', 'age', 'dribbling', 'passing']]
RW['RW_good'] =  np.where((pd.to_numeric(RW['pace']) >= 82) | (pd.to_numeric(RW['dribbling']) >= 73) | (pd.to_numeric(RW['passing']) >= 66) & (pd.to_numeric(RW['age']) <= 28), True, False)                         
RW_good_per =  round(RW['RW_good'].values.sum()/RW.shape[0], 3)


LW = df[df['player_positions'].str.contains('LW')].loc[:, ['pace', 'age', 'dribbling', 'passing']]
LW['LW_good'] =  np.where((pd.to_numeric(LW['pace']) >= 82) & (pd.to_numeric(LW['age']) <= 27)  | (pd.to_numeric(LW['dribbling']) >= 73) |  (pd.to_numeric(LW['passing']) >= 66), True, False)
LW_good_per =  round(LW['LW_good'].values.sum()/LW.shape[0], 3)

CAM = df[df['player_positions'].str.contains('CAM')].loc[:, ['pace', 'age', 'dribbling', 'passing']]
CAM['CAM_good'] =  np.where(( pd.to_numeric(CAM['pace']) >= 75 ) & (pd.to_numeric(CAM['age']) <= 28)  | (pd.to_numeric(CAM['dribbling']) >= 74)  | (pd.to_numeric(CAM['passing']) >= 70), True, False)
CAM_good_per =  round(CAM['CAM_good'].values.sum()/CAM.shape[0], 3)


CM = df[df['player_positions'].str.contains('CM')].loc[:, ['pace', 'age', 'dribbling', 'passing']]
CM['CM_good'] =  np.where((pd.to_numeric(CM['pace']) >= 70) & (pd.to_numeric(CM['age']) <= 28) | (pd.to_numeric(CM['dribbling']) >= 71) | ( pd.to_numeric(CM['passing']) >= 69), True,False)
CM_good_per =  round(CM['CM_good'].values.sum()/CM.shape[0], 3)


CDM = df[df['player_positions'].str.contains('CDM')].loc[:, ['pace', 'age', 'passing', 'physic']]
CDM['CDM_good'] =  np.where((pd.to_numeric(CDM['pace']) >= 68) & (pd.to_numeric(CDM['age']) <= 29 ) | (pd.to_numeric(CDM['passing']) >= 67) | (pd.to_numeric(CDM['physic']) >= 74), True, False)
CDM_good_per =  round(CDM['CDM_good'].values.sum()/CDM.shape[0], 3)

              
LWB = df[df['player_positions'].str.contains('LWB')].loc[:, ['pace', 'age', 'dribbling', 'physic']]
LWB['LWB_good'] =  np.where((pd.to_numeric(LWB['pace']) >= 78) & (pd.to_numeric(LWB['age']) <= 29)  | (pd.to_numeric(LWB['dribbling']) >= 69) | (pd.to_numeric(LWB['physic']) >= 71), True, False)
LWB_good_per =  round(LWB['LWB_good'].values.sum()/LWB.shape[0], 3)


RWB = df[df['player_positions'].str.contains('RWB')].loc[:, ['pace', 'age', 'dribbling', 'physic']]
RWB['RWB_good'] = np.where((pd.to_numeric(RWB['pace']) >= 79 ) & (pd.to_numeric(RWB['age']) <= 28) | ( pd.to_numeric(RWB['dribbling']) >= 70 ) | (pd.to_numeric(RWB['physic'] )>= 71),True,False)
RWB_good_per =  round(RWB['RWB_good'].values.sum()/RWB.shape[0], 3)


CB = df[df['player_positions'].str.contains('CB')].loc[:, ['pace', 'age', 'defending', 'physic']]
CB['CB_good'] =  np.where((pd.to_numeric(CB['pace']) >= 66) & ( pd.to_numeric(CB['age']) <= 29)  |  (pd.to_numeric(CB['defending']) >= 70) | (pd.to_numeric(CB['physic']) >= 75),True,False)
CB_good_per =  round(CB['CB_good'].values.sum()/CB.shape[0], 3)



LB = df[df['player_positions'].str.contains('LB')].loc[:, ['pace', 'age', 'dribbling', 'defending']]
LB['LB_good'] =  np.where((pd.to_numeric(LB['pace']) >= 76)  & ( pd.to_numeric(LB['age']) <= 29) | (pd.to_numeric(LB['defending']) >= 67) |( pd.to_numeric(LB['dribbling']) >= 68), True, False)  
LB_good_per =  round(LB['LB_good'].values.sum()/LB.shape[0], 3)

RB = df[df['player_positions'].str.contains('RB')].loc[:, ['pace', 'age', 'dribbling', 'defending']]
RB['RB_good'] = np.where((pd.to_numeric(RB['pace']) >= 77) & (pd.to_numeric(RB['age']) <= 28) | (pd.to_numeric(RB['dribbling']) >= 68) | (pd.to_numeric(RB['defending']) >= 68),True,False)
RB_good_per =  round(RB['RB_good'].values.sum()/RB.shape[0], 3)


def get_label(x):
    y = x.iloc[:,-1:]
    return y 



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(position):    
    X = position.iloc[:, :-1]
    y = position.iloc[:,-1:].values.ravel()
    bool_val = np.array(y)
    bool_val = np.multiply(bool_val, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)   
    #Following 80/20 rule, training and testing dataset is divided into train and test datasæt
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return [X, y, X_train, X_test, y_train, y_test]



from sklearn.linear_model import SGDClassifier

def SGD_logit_classifier(X_train, y_train):
    #Stochastic Gradient Descent as an optimisation algorithme to find the parameters vector, which minimising cost function
    clf = SGDClassifier(loss="log", max_iter=100).fit(X_train, y_train)
    return clf

from sklearn.linear_model import LogisticRegression

def lbfgs_clf(X_train, y_train):
    #Using lbfgs as an optimisation algorithme 
    clf =  LogisticRegression(multi_class='ovr', solver='lbfgs',penalty='l2', C=1.0,  tol=0.0001).fit(X_train, y_train)
    return clf

def mini_batchGD_clf(X_train, y_train):
    #match GD computes the gradients on small random sets of instances called minibatches
    clf = SGDClassifier(loss="log", max_iter=100).partial_fit(X_train, y_train, classes=np.unique(y_train))
    return clf
                        
                        
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def evaluate_model(classifier, X, y, X_test, y_test):
    #evaluting the model, output is absolute_error and accuracy 
    ##Cross validation for the model(k-n method)    
    cv = KFold(n_splits=10, random_state=1, shuffle=True)  
    scores = cross_val_score(classifier, X_test, y_test, scoring = "accuracy",
                             cv=cv, n_jobs=-1).mean() #use k-fold cross validation to evaluate the model, X, y are predicators and response variables, use accuracy scores to evalute model performance.        
    #Predict test results
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  
    #Use confusion matrix to calulate percentage of true positivity and true negativity in all y test dataset
    cm_accuracy = ( cm[0][0] + cm[1][1] ) / (len(y_test))
    print(scores, cm_accuracy)
    return scores, cm_accuracy

    
if __name__ == '__main__':

     X_train = preprocess_data(ST)[2]
     y_train = preprocess_data(ST)[4]
     X = preprocess_data(ST)[0]
     y = preprocess_data(ST)[1]
     X_test = preprocess_data(ST)[3]
     y_test = preprocess_data(ST)[5]
     clf = lbfgs_clf(X_train, y_train)
     sgd_clf = SGD_logit_classifier(X_train, y_train)
     mini_batch_clf = mini_batchGD_clf(X_train, y_train)
     evaluate_model(clf, X, y, X_test, y_test)
     evaluate_model(sgd_clf, X, y, X_test, y_test)
     evaluate_model(mini_batch_clf, X, y, X_test, y_test)
     ''' 
     y = get_label(ST)
     '''


        
        