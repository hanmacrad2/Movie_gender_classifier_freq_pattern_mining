"""
Author: Cliona O'Doherty

Description: A script for the analysis of labels returned from Amazon Rekognition video tagging. 
This script performs an association analysis on the labels, finding the items that frequently co-occur. 
Various association metrics are returned using mlxtend apriori and association rules
"""

import pickle
import glob
import pandas as pd

def create_baskets(dict1):
    """
    Create the empty baskets from the labels returned by Rekognition, one for each unique timestamp.
    """
    uniquetimestamps = set([x['Timestamp'] for x in dict1['alllabels']])
    basket = { i: [] for i in uniquetimestamps}
    return basket

def fill_baskets(dict1, dict2):
    """
    Fills the empty baskets with labels corresponding to the timestamp.
    dict1: the empty baskets for each movie (i.e. basket from create_baskets).
    dict2: the same as arg passed to create_baskets, i.e. each movie's labels.
    """
    for vals in dict2['alllabels']:
        dict1[vals['Timestamp']].extend([vals['Label']['Name']])
    
    return dict1

if __name__ == "__main__":

    #import labels    
    files = glob.glob('./data/*.pickle')
    movies = []
    for file in files:
        file_Name = file
        fileObject = open(file_Name, 'rb')
        file_labels = pickle.load(fileObject) 
        movies.append(file_labels)

    for movie in movies:
        del movie['compmsg']
        del movie['deltat']
        del movie['vid']

    #structure labels into "baskets" of latency 200 ms
    itemsets = []
    for movie in movies:
        baskets = fill_baskets(create_baskets(movie), movie)
        itemsets.extend(baskets.values())
    with open('itemsets.pickle', 'wb') as f:
        pickle.dump(itemsets, f)

    count = 0 
    for basket in itemsets:
        count += len(basket)
    print('The total number of baskets is {}'.format(len(itemsets)))
    print("The total number of labels is {}".format(count)) 

    