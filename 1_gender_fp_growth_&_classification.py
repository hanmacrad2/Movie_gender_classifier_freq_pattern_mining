# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:51:30 2020

@author: Hannah Craddock
"""

#Imports
import numpy as np
import pandas as pd
import itertools

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import (apriori,
                                       association_rules)
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier

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

def get_features(df_fp, df_ohe):
    '''Extract features X and y from dataframe of frequent itemsets. Only extract features that have occured with female or male'''
    
    #i.Get list of frequent items
    list_gender_freq_items = list(df_fp["itemsets_strings"])
    
    #ii. Extract from main one hot encoded dataframe
    df_ohe_gender_frs = df_ohe.loc[:, list_gender_freq_items]
    
    #Data
    X = df_ohe_gender_frs.drop(['Female'], axis=1)

    #Extract X, y
    y = df_ohe_gender_frs['Female']
    
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
redundant_labels_f = check_redundant(itemsets, 'Female')
df_fp_f = get_fp_gender(itemsets, gender, redundant_labels_f) 

X_f, y_f = get_features(df_fp_f, df_ohe)

#Get male features
redundant_labels_m = check_redundant(itemsets, 'Man')
df_fp_m = get_fp_gender(itemsets, gender, redundant_labels_m) 

X_m, y_m = get_features(df_fp_m, df_ohe)

#**************************************************************
#Modelling 
#Count of female == 1
#np.count_nonzero(ytest)

#************************************************************************************
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
choose_C_cv(X, y, c_range, plot_color)

#Final model (use default penalty term - no performance improvement for varying penalty)

def split_features(X,y):
    #Train + Test set
    testSizeX = 0.33 #67:33 split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= testSizeX, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

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

# Get train test splits for each gender
Xtrain_f, Xtest_f, ytrain_f, ytest_f = split_features(X_f, y_f)
Xtrain_m, Xtest_m, ytrain_m, ytest_m = split_features(X_m, y_m)

# Run the logistic regression model 
# i. Use the matching gender's features
run_logistic(Xtrain_f, Xtest_f, ytrain_f, ytest_f)
run_logistic(Xtrain_m, Xtest_m, ytrain_m, ytest_m)

# ii. Cross features to see if differences arise
run_logistic(Xtrain_f, Xtest_f, ytrain_m, ytest_m)
run_logistic(Xtrain_m, Xtest_m, ytrain_f, ytest_f)

#Auc


#************************************************
#SVM
def choose_C_SVM_cv(X, y, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_f1 =[]; std_f1 =[];
       
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
            #mse = mean_squared_error(y[test_index],ypred)
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

#*******************************************
#3. Baseline model
def run_dummy(Xtrain, Xtest, ytrain, ytest):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(Xtrain, ytrain)
    predictions_dummy = dummy_clf.predict(Xtest)

    #Evaluation
    print(confusion_matrix(ytest, predictions_dummy))
    print(classification_report(ytest, predictions_dummy))

# i. Use the matching gender's features
run_dummy(Xtrain_f, Xtest_f, ytrain_f, ytest_f)
run_dummy(Xtrain_m, Xtest_m, ytrain_m, ytest_m)

# ii. Cross features to see if differences arise
run_dummy(Xtrain_f, Xtest_f, ytrain_m, ytest_m)
run_dummy(Xtrain_m, Xtest_m, ytrain_f, ytest_f)


#*************************************************
#Compare performance - ROC curve

def plot_roc_models(Xtest, ytest, log_reg_model, knn_model, dummy_clf):
    'Plot ROC Curve of implemented models'
    
    #Logistic Regression model
    scores = log_reg_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, label = 'Logistic Regression')

    #knn model
    scores = knn_model.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores[:, 1])
    plt.plot(fpr,tpr, color = 'r', label = 'knn')

    #Baseline Model
    scores_bl = dummy_clf.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    plt.plot(fpr,tpr, color = 'orange', label = 'baseline model')
    
    #Random Choice
    plt.plot([0, 1], [0, 1],'g--') 

    #Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve') #  - Logistic Regression')

    plt.legend(['Logistic Regression', 'knn', 'Baseline ','Random Classifier'])
    plt.show()    
    
#Implement
plot_roc_models(Xtest, ytest, log_reg_model, knn_model, dummy_clf)

#Test
