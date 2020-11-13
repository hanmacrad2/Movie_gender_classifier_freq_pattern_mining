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

######################################################################
#Data
itemsets = pd.read_pickle("./itemsets.pickle")


#Inspect items
list_itemsets = [inner for outer in itemsets for inner in outer]
count_items = Counter(list_itemsets)
count_items.most_common()


#One hot encode list of lists
def get_df_items(itemsets):
    transaction_encoder = TransactionEncoder()
    transaction_encoded_ary = transaction_encoder.fit(itemsets).transform(itemsets)
    #Dataframe
    df = pd.DataFrame(transaction_encoded_ary, columns= transaction_encoder.columns_)
    return df 

df_ohe = get_df_items(itemsets)

##################################################################
#Frequent Patterns - For females

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
df_ohe_female = df_ohe.loc[:, list_female_freq_items]

#Data
X = df_ohe_female.drop(['Female'], axis=1)

#Extract X, y
y = df_ohe_female['Female']

#Model set up
#Data - Train + Test set
testSizeX = 0.33 #67:33 split
Xtrain, Xtest, ytrain, ytest, = train_test_split(X, y, test_size= testSizeX, random_state=42)
#Count of female == 1
np.count_nonzero(ytest)

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

