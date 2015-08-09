# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 22:00:43 2015

@author: gauravkhullar
"""


import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

directory = '/Users/gauravkhullar/Documents/MSLaptop_Backup/Personal/Aalto/Machine Learning/Excercise 2/'
file_name = 'training_data.txt'
cols = ['Decades','Exchange Rate']
full_data = pd.read_table(file_name,header=None,delim_whitespace=True,names=cols)

training_data = full_data.ix[0:37]
validation_data = full_data.ix[38:]

x_train = training_data['Exchange Rate']
Y_train = np.matrix(training_data['Decades'].as_matrix()).T

ones_vals = pd.Series(np.ones(50))
X_train = pd.DataFrame({'Coeff':ones_vals.ix[0:37],
               'Exchange Rates':x_train})
X_train = np.matrix(X_train.as_matrix())
#Betas_train = inv(X_train.T.dot(X_train)).dot(X_train.T.dot(Y_train))

first = inv(X_train.T.dot(X_train))
second = X_train.T.dot(Y_train)
Betas_first_order = np.matrix(first.dot(second))

#Compute validation error
x_valid = validation_data['Exchange Rate']
Y_valid = np.matrix(validation_data['Decades'].as_matrix()).T
X_valid = pd.DataFrame({'Coeff':ones_vals.ix[38:],
               'Exchange Rates':x_valid})

X_valid = np.matrix(X_valid.as_matrix())
sq_error = np.array(Y_valid - (X_valid.dot(Betas_first_order))) ** 2

mean_error = sq_error.sum()/sq_error.size
#mean_sq_error 