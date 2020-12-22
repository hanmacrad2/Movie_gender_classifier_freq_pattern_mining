# Movie_gender_classifier_freq_pattern_mining
A repository for the project submitted by Hannah Craddock and Cliona O'Doherty using machine learning to investigate gender representations in a movie dataset.

## Data preprocessing
[create_itemsets.py](https://github.com/hanmacrad2/Movie_gender_classifier_freq_pattern_mining/blob/main/create_itemsets.py) - retrieve the raw output files from Rekognition (contact odoherc1@tcd.ie for access) and process them into an array with each row containing item strings which occurred in the same 200 ms interval. 

## Modelling
[gender_modelling.py](https://github.com/hanmacrad2/Movie_gender_classifier_freq_pattern_mining/blob/main/gender_modelling.py) - script for the classification analysis of two genders: male and female. This script takes the preprocessed data from create_itemsets and performs the following:
  (i) performs an association analysis using fpgrowth to find the most associated objects with each gender
  (ii) generates features from these objects
  (iii) inputs these into a linear classifier (logistic regression, SVM, baseline)
  (iv) performs evaluation including confusion matrices and ROC plots

## Results
Frequent pattern mining results can be found in [./feature_results](https://github.com/hanmacrad2/Movie_gender_classifier_freq_pattern_mining/tree/main/feature_results)

ROC plots are in [./roc_plots](https://github.com/hanmacrad2/Movie_gender_classifier_freq_pattern_mining/tree/main/roc_plots). These include ROCs for each gender, for both compare and for gender cross results (using female features to predict male).

[itemsets_eg.txt](https://github.com/hanmacrad2/Movie_gender_classifier_freq_pattern_mining/blob/main/itemsets_eg.txt) illustrates a small sample of the preprocessed dataset returned from create_itemsets.py. Should you wish to access the entire dataset, please contact odoherc1@tcd.ie





