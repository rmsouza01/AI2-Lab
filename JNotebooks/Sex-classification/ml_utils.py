import numpy as np
import pandas as pd

def get_folds(data,test_split, to_drop, to_keep):
    test_sets = []

    for ii in test_split:
        txt_file = open(ii)   
        test_aux = []
        for id in txt_file:
            test_aux.append(id.strip())
        test_aux2 = data[data['Measure_volume'].isin(test_aux)]
        test_sets.append(test_aux)

    #Iterating through each test set
    for test_set in test_sets:
        train_set = data[~data['Measure_volume'].isin(test_set)] #Get the train set by getting all the records excluding test records
        Xtrain = train_set.drop(to_drop, axis = 1) #Drop irrelevant columns for training from training feature set
        Ytrain = train_set[to_keep] #Getting gender labels
        test_set = data[data['Measure_volume'].isin(test_set)]
        Xtest = test_set.drop(to_drop, axis = 1) #Drop irrelevant columns for training from testing feature set
        Ytest = test_set[to_keep] #Getting gender labels for accuracy calculation
        yield (Xtrain,Ytrain,Xtest,Ytest)
