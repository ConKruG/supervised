#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 00:01:28 2018

@author: freeze
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import time
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Start and End date of the dataset
#start_date = '2009-01-05'
#end_date   = '2018-02-09'

# Path of the data file (Must be changed according to where you put your file!)
#path       = '/home/freeze/Documents/DiMex/newdata.xls'



start_date = '2007-09-17'
end_date   = '2015-06-04'
#path       = '/home/arash/Documents/dataset.xls'
#path       = '/home/freeze/Documents/NetPaper/dataset.xls'
path       = '/home/freeze/Documents/NetPaper/dataset2.xls'



# proportion of train and validation in the whole dataset
# proportion_of_test => 1 - prop_train - prop_val
prop_train, prop_val = 0.7, 0.2

# 31 days before to predict next day for supervised prediction methods
delay = 32
# Known nodes in the network = Known time series that reflect other time series futures
known_nodes = 9

rs = 37 #Random State


###############################################################################        
# Data Preparation Block
# Input: an excel file of markets time series, start and end date as string with format of YYYY-MM-DD
# Output: matrix of markets log-return time series (rows as days and columns as markets) in numpy array and names of markets in headers list
        
def data_reader(path, start_date, end_date):
    
    #log("data_reader()...")
    
    xl = pd.read_excel(path)
    xl['Date'] = pd.to_datetime(xl['Date'])
    xl = xl.set_index('Date')
    xl = xl.loc[start_date:end_date]
    xl = xl.dropna()
    
    #Making the log return series out of pure values of indices and commodity prices
    xl = np.log(xl/xl.shift(1))
    
    xl = xl.dropna()
    
    dataset = xl.values
    headers = list(xl.columns)
    
    return [dataset, headers]
        
###############################################################################
# Splits the dataset into train, validation and test datasets
# Input: dataset!, proportion for train, proportion for validation
# Output: train, validation, and test datasets in numpy array format

def train_validation_test_maker(dataset, prop_train, prop_val):
    
    #log('train_validation_test_maker()...')
    
    rows = dataset.shape[0]
    rows_train = int(rows * prop_train)
    rows_val   = int(rows * prop_val)
    
    train = dataset[:rows_train, :].copy()
    val   = dataset[rows_train:rows_train + rows_val, :].copy()
    test  = dataset[rows_train + rows_val:, :].copy()
    
    return [train, val, test]

     
###############################################################################
# Changes a time series into a dataset of n days before (delay) and next day
# Input: A time series and how many days before is important
# Output: delayed dataset
    
def delay_maker(series, delay = delay):
    
    l = len(series)
    new_data = []
    for i in range(0, l - delay):
        temp = series[i: i + delay + 1]
        new_data.append(temp)
        
    new_data = np.array(new_data)
    
    new_data[:, -1] = 1 * (new_data[:, -1] > 0)
    
    return new_data

###############################################################################

[dataset, headers] = data_reader(path, start_date, end_date)

[train, validation, test] = train_validation_test_maker(dataset, prop_train, prop_val)

# Train and validation together for the test phase (After validation phase)
trval = np.vstack((train, validation))


num_of_markets = trval.shape[1]

#SVM, RF, DT, KNN, NB, ANN, CNN

from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

results = np.zeros(((num_of_markets - known_nodes), 7))

for i in range(known_nodes, num_of_markets):
    
    series = dataset[:, i]
    
    data = delay_maker(series)
    
    [data_train, data_validation, data_test] = train_validation_test_maker(data, prop_train, prop_val)
    
    data_trval = np.vstack((data_train, data_validation))
    
    
    ### SVM
    
    SVM = svm.SVC(C = 10000, gamma = 0.01, random_state = rs)
    SVM = SVM.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = SVM.predict(data_test[:, :-1])
    
    print(i - known_nodes, "SVM")
    results[i - known_nodes, 0] = accuracy_score(data_test[:, -1], y_pred)
    
    ### RANDOM FOREST
    
    RF = RandomForestClassifier(n_estimators = 100, random_state = rs)
    RF = RF.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = RF.predict(data_test[:, :-1])
    
    print(i - known_nodes, "RF")
    results[i - known_nodes, 1] = accuracy_score(data_test[:, -1], y_pred)
    
    
    ### DECISION TREE 
    
    DT = tree.DecisionTreeClassifier(random_state = rs)
    DT = DT.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = DT.predict(data_test[:, :-1])
    
    print(i - known_nodes, "DT")
    results[i - known_nodes, 2] = accuracy_score(data_test[:, -1], y_pred)

    ### KNN
    
    KNN = KNeighborsClassifier()
    KNN = KNN.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = KNN.predict(data_test[:, :-1])
    
    print(i - known_nodes, "KNN")
    results[i - known_nodes, 3] = accuracy_score(data_test[:, -1], y_pred)
    
    ### NB
    
    NB = GaussianNB()
    NB = NB.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = NB.predict(data_test[:, :-1])
    
    print(i - known_nodes, "NB")
    results[i - known_nodes, 4] = accuracy_score(data_test[:, -1], y_pred)

    ### MLP
    
    #MLP = MLPClassifier(hidden_layer_sizes=(100,), random_state = rs)
    MLP = MLPClassifier(hidden_layer_sizes=(10,), random_state = rs)
    MLP = MLP.fit(data_trval[:, :-1], data_trval[:, -1])
    
    y_pred = MLP.predict(data_test[:, :-1])
    
    print(i - known_nodes, "MLP")
    results[i - known_nodes, 5] = accuracy_score(data_test[:, -1], y_pred)
        
final_results = np.mean(results, axis = 0)

    
