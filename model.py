# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:56:32 2020

@author: NIKHIL
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0,inplace = True)

dataset['test_score'].fillna(dataset['test_score'].mean(),inplace = True)

X = dataset.iloc[:,:3]

def convert_to_int(word):
    word_dict = {'one'    : 1,
                 'two'    : 2,
                 'three'  : 3,
                 'four'   : 4,
                 'five'   : 5,
                 'six'    : 6,
                 'seven'  : 7,
                 'eight'  : 8,
                 'nine'   : 9,
                 'ten'    : 10,
                 'eleven' : 11,
                 'twelve' : 12,
                 'zero'   : 0,
                 '0'      : 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[9,6,2]]))

