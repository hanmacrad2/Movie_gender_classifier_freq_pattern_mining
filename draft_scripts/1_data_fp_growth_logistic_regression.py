# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:05:00 2020

- Extract frequent items in clips that contain female

- Use these as input features (X) to predict gender (y) across all clips (including those without female)
 
@author: Hannah Craddock
"""
#Imports
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import (apriori,
                                       association_rules)
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import sklearn
#************************************************
#i. Data
itemsets = pd.read_pickle("./itemsets.pickle")

#Inspect items
list_itemsets = [inner for outer in itemsets for inner in outer]
#count_items = Counter(list_itemsets)
#count_items.most_common()

#One hot encode data (list of lists) 
def get_df_items(itemsets):
    '''Creates a one-hot encoded dataframe of the given itemsets'''
    transaction_encoder = TransactionEncoder()
    transaction_encoded_ary = transaction_encoder.fit(itemsets).transform(itemsets)
    #Dataframe
    df = pd.DataFrame(transaction_encoded_ary, columns= transaction_encoder.columns_)
    return df 

df_ohe = get_df_items(itemsets)

#************************************************
#Frequent Patterns - For females
def get_fp_gender(itemsets, gender):
    '''Get model input X and y for a given gender.  '''
    
    #i. Extract the itemsets for which the given gender is present
    itemsets_gender = [x for x in itemsets if gender in x]   
    
    #ii. Create ohe dataframe of items that contain the gender  
    df_gender = get_df_items(itemsets_gender) 
    
    #iii. Frequent Patterns - FP Growth
    df_fp = fpgrowth(df_gender, min_support= 0.01, max_len = 1, use_colnames=True)
    
    #iv.Extract strings from frozen sets
    df_fp["itemsets_strings"] = df_fp["itemsets"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    
    return df_fp

#Female
#Female clips
#1. Fiter for the occurence of female in a list
filterX = 'Female'
itemsets_female = [x for x in itemsets if filterX in x]   
#Create ohe dataframe of items that contain female 
df_female = get_df_items(itemsets_female) 
#2. Frequent Patterns - FP Growth - for female
df_female_fp = fpgrowth(df_female, min_support= 0.01, max_len = 1, use_colnames=True)

#Extract features as a list
#i. Convert frozen sets to strings 
df_female_fp["itemsets_strings"] = df_female_fp["itemsets"].apply(lambda x: ', '.join(list(x))).astype("unicode")
#ii. Convert to list
list_female_freq_items = list(df_female_fp["itemsets_strings"])

##################################################################
#Subset features across all itemsets for Model
df_ohe_female_frs = df_ohe.loc[:, list_female_freq_items]

#Data
X = df_ohe_female_frs.drop(['Female'], axis=1)

#Extract X, y
y = df_ohe_female_frs['Female']

#Model set up
#Data - Train + Test set
testSizeX = 0.33 #67:33 split
Xtrain, Xtest, ytrain, ytest, = train_test_split(X, y, test_size= testSizeX, random_state=42)
#Count of female == 1
#np.count_nonzero(ytest)

#************************************************************************************
#Model 1 - Logistic Regression

#Logistic Regression Classifier
log_reg_model = LogisticRegression(penalty= 'l2')
log_reg_model.fit(Xtrain, ytrain)
log_reg_model.intercept_
log_reg_model.coef_

#Predictions
predictions = log_reg_model.predict(Xtest)

#***********
#Performance
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))

#Choose best c
#i. Choose c
def choose_C_cv(X, y, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_f1 =[]; std_f1 =[];
       
    #Loop through each k fold
    for c_param in c_range:
        f1_temp = [] 
        model = LogisticRegression(penalty= 'l2', C = c_param)
                
        for train_index, test_index in kf.split(X):           

            model.fit(X.iloc[list(train_index)], y[train_index])
            ypred = model.predict(X.iloc[list(test_index)])
            f1X = f1_score(y[test_index],ypred)
            #mse = mean_squared_error(y[test_index],ypred)
            f1_temp.append(f1X)
        
        #Get mean & variance
        mean_f1.append(np.array(f1_temp).mean())
        std_f1.append(np.array(f1_temp).std())
        
    #Plot
    plt.errorbar(c_range, mean_f1, yerr=std_f1, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean F1 score')
    plt.title('Choice of C in Logistic regression - 5 fold CV')
    plt.show()
    
#Implement
c_range = [0.001, 0.01, 1, 10, 30, 50, 100, 500, 1000]
plot_color = 'g'
choose_C_cv(X, y, c_range, plot_color)

#Performance
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))

#Roc 

#Auc




#************************************************
#knn 
#i. Choose k
def choose_k_knn(X, y, k_range, plot_color):
    '''knn - Implement 5 fold cross validation for determinine optimal k'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_error=[]; std_error=[];
       
    #Loop through each k fold
    for k in k_range:
        mse_temp = []
        model = KNeighborsClassifier(n_neighbors = k, weights= 'uniform')
                
        for train, test in kf.split(X):
            
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            mse = mean_squared_error(y[test],ypred)
            mse_temp.append(mse)
        
        #Get mean & variance
        mean_error.append(np.array(mse_temp).mean())
        std_error.append(np.array(mse_temp).std())
        
    #Plot
    plt.errorbar(k_range, mean_error, yerr=std_error, color = plot_color)
    plt.xlabel('k')
    plt.ylabel('Mean square error')
    plt.title('kNN - 5 fold CV')
    plt.show()

#Implement
k_range = [2,3,5,7,8,10,15,20,25,40, 60, 100]
plot_color = 'orange'
choose_k_knn(X, y, k_range, plot_color)    

#*******************************************
#3. Baseline model

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X[indices_train], y[indices_train])
predictions_dummy = dummy_clf.predict(X[indices_test])
#Confusion matrix
print(confusion_matrix(ytest, predictions_dummy))
