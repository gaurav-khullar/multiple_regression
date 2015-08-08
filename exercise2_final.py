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
ones_vals = pd.Series(np.ones(50))


               
def regress_coeffs(X_train,Y_train) :
    first = inv(X_train.T.dot(X_train))
    second = X_train.T.dot(Y_train)
    Betas_n_order = first.dot(second)
    return Betas_n_order

def regress_error(X_valid,Y_valid,Betas_first_order):
    error = Y_valid - (X_valid.dot(Betas_first_order))
    sq_error = error ** 2
    mean_error = sq_error.sum()/sq_error.size
    return mean_error

#First Order

X_train_first = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train})
X_valid_first = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid})
Betas_first_order = regress_coeffs(X_train_first,Y_train)
error_first_order = regress_error(X_valid_first,Y_valid,Betas_first_order)
#Compute validation error

#Second Order
X_train_second = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2})
X_valid_second = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2})

Betas_sec_order = regress_coeffs(X_train_second,Y_train)
error_sec_order = regress_error(X_valid_second,Y_valid,Betas_sec_order)

#3rd order
X_train_third = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2,
               'Decades_3':x_train ** 3})
X_valid_third = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2,
               'Decades_3':x_valid ** 3})

Betas_third_order = regress_coeffs(X_train_third,Y_train)
error_third_order = regress_error(X_valid_third,Y_valid,Betas_third_order)

#4th Order
X_train_fourth = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2,
               'Decades_3':x_train ** 3,
               'Decades_4':x_train ** 4})
X_valid_fourth = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2,
               'Decades_3':x_valid ** 3,
               'Decades_4':x_valid ** 4})
Betas_fourth_order = regress_coeffs(X_train_fourth,Y_train)
error_fourth_order = regress_error(X_valid_fourth,Y_valid,Betas_fourth_order)

#5th Order
X_train_fifth = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2,
               'Decades_3':x_train ** 3,
               'Decades_4':x_train ** 4,
               'Decades_5':x_train ** 5})
X_valid_fifth = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2,
               'Decades_3':x_valid ** 3,
               'Decades_4':x_valid ** 4,
               'Decades_5':x_valid ** 5})
Betas_fifth_order = regress_coeffs(X_train_fifth,Y_train)
error_fifth_order = regress_error(X_valid_fifth,Y_valid,Betas_fifth_order)

#6th Order
X_train_sixth = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2,
               'Decades_3':x_train ** 3,
               'Decades_4':x_train ** 4,
               'Decades_5':x_train ** 5,
               'Decades_6':x_train ** 6})
X_valid_sixth = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2,
               'Decades_3':x_valid ** 3,
               'Decades_4':x_valid ** 4,
               'Decades_5':x_valid ** 5,
               'Decades_6':x_valid ** 6})
Betas_sixth_order = regress_coeffs(X_train_sixth,Y_train)
error_sixth_order = regress_error(X_valid_sixth,Y_valid,Betas_sixth_order)

#7th Order

X_train_seventh = pd.DataFrame({'Coeff':ones_vals.ix[0:30],
               'Decades':x_train,
               'Decades_2':x_train ** 2,
               'Decades_3':x_train ** 3,
               'Decades_4':x_train ** 4,
               'Decades_5':x_train ** 5,
               'Decades_6':x_train ** 6,
               'Decades_7':x_train ** 7})
X_valid_seventh = pd.DataFrame({'Coeff':ones_vals.ix[31:],
               'Decades':x_valid,
               'Decades_2':x_valid ** 2,
               'Decades_3':x_valid ** 3,
               'Decades_4':x_valid ** 4,
               'Decades_5':x_valid ** 5,
               'Decades_6':x_valid ** 6,
               'Decades_7':x_valid ** 7})
Betas_seventh_order = regress_coeffs(X_train_seventh,Y_train)
error_seventh_order = regress_error(X_valid_seventh,Y_valid,Betas_seventh_order)
####
#Errors
errors = np.array([error_first_order,error_sec_order,error_third_order,error_fourth_order,
                  error_fifth_order,error_sixth_order,error_seventh_order])
betas = pd.DataFrame([Betas_first_order,Betas_sec_order,Betas_third_order,Betas_fourth_order,
                  Betas_fifth_order,Betas_sixth_order,Betas_seventh_order],index=range(1,8))




#PLots
plt.scatter(x_valid,Y_valid)
plt.autoscale(tight=True)
plt.grid()
plt.title("Exchange Rates versus Decades (1999-2013)")
plt.xlabel("Decades")
plt.ylabel("Exchange Rate Eur/Dollar")
plt.plot(x_valid,X_valid_first.dot(Betas_first_order), color='red',linewidth=4)
plt.plot(x_valid,X_valid_second.dot(Betas_sec_order), color='blue',linewidth=4)
plt.plot(x_valid,X_valid_third.dot(Betas_third_order), color='green',linewidth=4)
plt.plot(x_valid,X_valid_fourth.dot(Betas_fourth_order), color='black',linewidth=4)
#plt.plot(x_valid,X_valid_fifth.dot(Betas_fifth_order), color='m',linewidth=4)
plt.legend(["d = 1","d = 2","d = 3","d = 4"],loc = "upper left")
plt.show()

print errors


############ Testing on Test Set #########
test_data = pd.read_table(test_data_file,header=None,delim_whitespace=True,names=cols)
test_data = test_data.sort_index(by='Decades')
test_data.index = range(0,len(test_data))
x_test = test_data['Decades']
Y_test = test_data['Exchange Rate']
ones_vals_test = pd.Series(np.ones(len(test_data)))


X_test_first = pd.DataFrame({'Coeff':ones_vals_test.ix[:],
               'Decades':x_test})
               
X_test_second = pd.DataFrame({'Coeff':ones_vals_test.ix[:],
               'Decades':x_test,
               'Decades_2':x_test ** 2})
               
X_test_third = pd.DataFrame({'Coeff':ones_vals_test.ix[:],
               'Decades':x_test,
               'Decades_2':x_test ** 2,
               'Decades_3':x_test ** 3})
               
X_test_seventh = pd.DataFrame({'Coeff':ones_vals_test.ix[0:30],
               'Decades':x_test,
               'Decades_2':x_test ** 2,
               'Decades_3':x_test ** 3,
               'Decades_4':x_test ** 4,
               'Decades_5':x_test ** 5,
               'Decades_6':x_test ** 6,
               'Decades_7':x_test ** 7})

error_first_order_test = regress_error(X_test_first,Y_test,Betas_first_order)
error_second_order_test = regress_error(X_test_second,Y_test,Betas_sec_order)
error_third_order_test = regress_error(X_test_third,Y_test,Betas_third_order)
error_seventh_order_test = regress_error(X_test_seventh,Y_test,Betas_seventh_order)

plt.scatter(x_test,Y_test)
plt.autoscale(tight=True)
plt.grid()
plt.title("TEST DATA - Exchange Rates versus Decades (1999-2013)")
plt.xlabel("Decades")
plt.ylabel("Exchange Rate Eur/Dollar")

plt.plot(x_test,X_test_first.dot(Betas_first_order), color='yellow',linewidth=4)
plt.plot(x_test,X_test_second.dot(Betas_sec_order), color='green',linewidth=4)
plt.plot(x_test,X_test_third.dot(Betas_third_order), color='red',linewidth=4)
#plt.plot(x_test,X_test_seventh.dot(Betas_seventh_order), color='yellow',linewidth=4)
plt.legend(["d = 1","d = 2","d = 3"],loc = "upper left")
plt.show()