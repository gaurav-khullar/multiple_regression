# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 22:00:43 2015

@author: gauravkhullar
"""


import pandas as pd
import matplotlib.pyplot as plt
from regression import *

file_name = 'training_data.txt'
test_data_file = 'test_data.txt'
cols = ['Decades','Exchange Rate']
full_data = pd.read_table(file_name,header=None,delim_whitespace=True,names=cols)
full_data = full_data.sort_index(by='Decades')
full_data.index = range(0,len(full_data))

training_data = full_data.ix[0:30]
validation_data = full_data.ix[31:]

x_train = training_data['Decades']
Y_train = training_data['Exchange Rate']
x_valid = validation_data['Decades']
Y_valid = validation_data['Exchange Rate']

#First Order

#X_train_first = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
#               'Decades':x_train})
#X_valid_first = pd.DataFrame({'Coeff':ones_vals.ix[31:],
#               'Decades':x_valid})

first_order = Regression(x_train,Y_train,x_valid,Y_valid,1) 
first_order.linear_basis_function()
first_order.coeffs()
first_order.empirical_error()
first_order.generalization_error()
#Compute validation error
