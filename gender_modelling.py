# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:51:30 2020

@author: Hannah Craddock, Cliona O'Doherty
"""

#Imports
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import (apriori,
                                       association_rules)
from collections import Counter

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC

#************************************************
#i. Data
itemsets = pd.read_pickle("./itemsets.pickle")

#Inspect items
list_itemsets = [inner for outer in itemsets for inner in outer]
#count_items = Counter(list_itemsets)
#count_items.most_common()

#****************************************************************
#Frequent Pattern Mining

#i. One hot encode data (list of lists) 
def get_df_items(itemsets):
    '''Creates a one-hot encoded dataframe of the given itemsets'''
    transaction_encoder = TransactionEncoder()
    transaction_encoded_ary = transaction_encoder.fit(itemsets).transform(itemsets)
    #Dataframe
    df = pd.DataFrame(transaction_encoded_ary, columns= transaction_encoder.columns_)
    return df 

#ii. Frequent Pattern Mining 
def get_fp_gender(itemsets, gender, redundant_labels):
    '''Get model input X and y for a given gender.  '''
    
    #i. Extract the itemsets for which the given gender is present
    itemsets_genderI = [inner_items for inner_items in itemsets if gender in inner_items]   
    
    #ii. Remove redundant labels
    itemsets_gender = [[item for item in inner_items if item not in redundant_labels] for inner_items in itemsets_genderI]  
        
    #iii. Create ohe dataframe of items that contain the gender  
    df_gender = get_df_items(itemsets_gender) 
    
    #iii. Frequent Patterns - FP Growth
    df_fp = fpgrowth(df_gender, min_support= 0.01, max_len = 1, use_colnames=True)
    
    #iv.Extract strings from frozen sets
    df_fp["itemsets_strings"] = df_fp["itemsets"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    
    return df_fp

def get_features(df_fp, df_ohe, gender):
    '''Extract features X and y from dataframe of frequent itemsets. Only extract features that have occured with female or male'''
    #i.Get list of frequent items
    list_gender_freq_items = list(df_fp["itemsets_strings"])
    
    #ii. Extract from main one hot encoded dataframe
    df_ohe_gender_frs = df_ohe.loc[:, list_gender_freq_items]
    
    #Data
    X = df_ohe_gender_frs.drop([gender], axis=1)

    #Extract X, y
    y = df_ohe_gender_frs[gender]
    
    return X, y 

def check_redundant(l, ref):
    """Find the elements of l which always occur with word ref"""
    
    #i. Condense l to only those elements containing ref
    l = [i for i in l if ref in i]

    #ii. Define valid which checks whether two words in an element of l always occur together
    def valid(p):
        for s in l:
            if any(e in s for e in p) and not all(e in s for e in p):
                return False
            return True
    
    #iii. Find unique words
    elements = list(set(b for a in l for b in a))

    #iv. Check all pairs of combinations and store in a list to return
    pairs = []
    for c in itertools.combinations(elements, 2):
        if ref in c:
            if valid(c):
                pairs.append(c)
    
    pairs = list(set(b for a in pairs for b in a))
    pairs.remove(ref)

    return pairs
    
#Apply - get ohe dataframe of itemsets
df_ohe = get_df_items(itemsets)

#Get female features

#use check_redundant to get all labels that when they appear, always appear with 'Female'
redundant_labels_f = check_redundant(itemsets, 'Female')

#Manually elect the synonymous or uninformative labels that occur redundantly with 'Female' (from check_redundant)
redundant_labels_f = ['Person','Face','Woman','Human','Indoors','Head']
df_fp_f = get_fp_gender(itemsets, 'Female', redundant_labels_f) 

X_f, y_f = get_features(df_fp_f, df_ohe, 'Female')

#Get male features
#redundant_labels_m = check_redundant(itemsets, 'Man')
redundant_labels_m = ['Person','Face','Human','Indoors','Head']
df_fp_m = get_fp_gender(itemsets, 'Man', redundant_labels_m) 

X_m, y_m = get_features(df_fp_m, df_ohe, 'Man')

# Get train test splits for each gender
testSizeX = 0.33 #67:33 split
Xtrain_f, Xtest_f, ytrain_f, ytest_f = train_test_split(X_f, y_f, test_size= testSizeX, random_state=42)
Xtrain_m, Xtest_m, ytrain_m, ytest_m = train_test_split(X_m, y_m, test_size= testSizeX, random_state=42)

#**************************************************************
#Modelling 

#*******************************************************************************************************
#Model 1 - Logistic Regression

#i. Choose c
def choose_C_cv(X, y, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_f1 =[]; std_f1 =[]
       
    #Loop through each k fold
    for c_param in c_range:
        print('C = {}'.format(c_param))
        count = 0; f1_temp = [] 
        model = LogisticRegression(penalty= 'l2', C = c_param)
                
        for train_index, test_index in kf.split(X):           
            count = count + 1 
            print('count kf = {}'.format(count))
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

#i. get female cv results
choose_C_cv(X_f, y_f, c_range, plot_color)
#ii. get male cv results
choose_C_cv(X_m, y_m, c_range, plot_color)

#Final model (use default penalty term - no performance improvement for varying penalty)
def run_logistic(Xtrain, Xtest, ytrain, ytest):
    log_reg_model = LogisticRegression(penalty= 'l2')
    log_reg_model.fit(Xtrain, ytrain)

    #log_reg_model.intercept_
    #log_reg_model.coef_
    #Predictions
    predictions = log_reg_model.predict(Xtest)

    #Performance
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))
    
    #Auc
    scores = log_reg_model.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores[:, 1])
    print('AUC = {}'.format(auc(fpr, tpr)))

    return log_reg_model

# Run the logistic regression model 
# i. Use the matching gender's features
log_reg_model_f = run_logistic(Xtrain_f, Xtest_f, ytrain_f, ytest_f)
log_reg_model_m = run_logistic(Xtrain_m, Xtest_m, ytrain_m, ytest_m)

#*********************************************************************************************************
#SVM
def choose_C_SVM_cv(X, y, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_f1 =[]; std_f1 =[]
       
    #Loop through each k fold
    for c_param in c_range:
        print('C = {}'.format(c_param))
        count = 0; f1_temp = [] 
        model = LinearSVC(C = c_param)
                
        for train_index, test_index in kf.split(X):           
            count = count + 1 
            print('count kf = {}'.format(count))
            model.fit(X.iloc[list(train_index)], y[train_index])
            ypred = model.predict(X.iloc[list(test_index)])
            f1X = f1_score(y[test_index],ypred)
            f1_temp.append(f1X)
        
        #Get mean & variance
        mean_f1.append(np.array(f1_temp).mean())
        std_f1.append(np.array(f1_temp).std())
        
    #Plot
    plt.errorbar(c_range, mean_f1, yerr=std_f1, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean F1 score')
    plt.title('Choice of penatly term C in SVM - 5 fold CV')
    plt.show()

#Implement
c_range = [0.001, 0.01, 1, 10, 100]
plot_color = 'g' 

#i. female cv results
choose_C_SVM_cv(X_f, y_f, c_range, plot_color)
#i. male cv results
choose_C_SVM_cv(X_m, y_m, c_range, plot_color)

def run_svm(Xtrain, Xtest, ytrain, ytest, c_param=1.0):
    svm_model = LinearSVC(C = c_param)
    svm_model.fit(Xtrain, ytrain)

    #Predictions
    predictions = svm_model.predict(Xtest)

    #Performance
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))
    
    #Auc
    scores = svm_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    print('AUC = {}'.format(auc(fpr, tpr)))

    return svm_model

# Run svm model - Use C=1.0 (default) as per cross val results
# i. Use the matching gender's features
svm_model_f = run_svm(Xtrain_f, Xtest_f, ytrain_f, ytest_f)
svm_model_m = run_svm(Xtrain_m, Xtest_m, ytrain_m, ytest_m)

#*******************************************
#3. Baseline model
def run_dummy(Xtrain, Xtest, ytrain, ytest):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(Xtrain, ytrain)
    predictions_dummy = dummy_clf.predict(Xtest)

    #Evaluation
    print(confusion_matrix(ytest, predictions_dummy))
    print(classification_report(ytest, predictions_dummy))
    
    #Auc
    scores_bl = dummy_clf.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    print('AUC = {}'.format(auc(fpr, tpr)))

    return dummy_clf

# i. Use the matching gender's features
dummy_clf_f = run_dummy(Xtrain_f, Xtest_f, ytrain_f, ytest_f)
dummy_clf_m = run_dummy(Xtrain_m, Xtest_m, ytrain_m, ytest_m)

#*************************************************
#Compare performance - ROC curve

def plot_roc_models(Xtest, ytest, log_reg_model, svm_model, dummy_clf, gender=''):
    'Plot ROC Curve of implemented models'
    
    #Logistic Regression model
    scores = log_reg_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, label = 'Logistic Regression')
    print('AUC = {}'.format(auc(fpr, tpr)))

    #svm model
    scores = svm_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, color = 'r', label = 'svm')
    print('AUC = {}'.format(auc(fpr, tpr)))

    #Baseline Model
    scores_bl = dummy_clf.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    plt.plot(fpr,tpr, color = 'orange', label = 'baseline model')
    print('AUC = {}'.format(auc(fpr, tpr)))
    
    #Random Choice
    plt.plot([0, 1], [0, 1],'g--') 

    #Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve - {}'.format(gender))

    plt.legend(['Logistic Regression', 'SVM', 'Baseline (most freq)','Random Classifier']) 
    plt.savefig('./roc_{}'.format(gender))
    plt.show()
    plt.close()
    
#Implement
plot_roc_models(Xtest_f, ytest_f, svm_model_f, log_reg_model_f, dummy_clf_f, gender='Female')
plot_roc_models(Xtest_m, ytest_m, svm_model_m, log_reg_model_m, dummy_clf_m, gender='Male')

#********************************
#Compare ROC curves


#Logistic Regression model - matched features
scores = log_reg_model_f.decision_function(Xtest_f)
fpr, tpr, _= roc_curve(ytest_f, scores)
plt.plot(fpr,tpr, label = 'Logistic Regression - female')
print('AUC = {}'.format(auc(fpr, tpr)))

#Logistic Regression model - crossed features
scores = log_reg_model_m.decision_function(Xtest_m)
fpr, tpr, _= roc_curve(ytest_m, scores)
plt.plot(fpr,tpr, label = 'Logistic Regression - male')
print('AUC = {}'.format(auc(fpr, tpr)))

#Baseline Model
scores_bl = dummy_clf_f.predict_proba(Xtest_f)
fpr, tpr, _= roc_curve(ytest_f, scores_bl[:, 1])
plt.plot(fpr,tpr, label = 'Baseline model - female')
print('AUC = {}'.format(auc(fpr, tpr)))

#Baseline Model
scores_bl = dummy_clf_m.predict_proba(Xtest_m)
fpr, tpr, _= roc_curve(ytest_m, scores_bl[:, 1])
plt.plot(fpr,tpr, label = 'Baseline model - male')
print('AUC = {}'.format(auc(fpr, tpr)))

#Random Choice
plt.plot([0, 1], [0, 1],'g--') 

#Labels
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve - logistic regression male vs. female')

plt.legend() 
plt.savefig('./roc_comp')
plt.show()
plt.close()

