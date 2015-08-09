# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:10:20 2015

@author: gauravkhullar
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv

class Regression:
    def __init__(self, x_vals, y_vals, x_valid, y_valid, order):
        self.X_vals_trg = x_vals
        self.Y_vals_trg = y_vals
        self.X_vals_valid = x_valid
        self.Y_vals_valid = y_valid
        self.Y_vals_valid.index = range(0,len(self.Y_vals_valid))
        self.order = order
        self.betas=[]
        self.emp_error=0.0
        self.gen_error = 0.0
        self.basis_function_trg = pd.DataFrame()
        self.basis_function_valid = pd.DataFrame()
        
        
    def linear_basis_function(self):
        ones_vals_trg = pd.Series(np.ones(len(self.X_vals_trg)))   
        ones_vals_valid = pd.Series(np.ones(len(self.X_vals_valid))) 
        self.basis_function_trg = pd.DataFrame(ones_vals_trg,columns = ['w0'])
        self.basis_function_valid = pd.DataFrame(ones_vals_valid,columns = ['w0'])
        i = 1
        while i<= self.order:
            new_column_trg = pd.Series(np.array(self.X_vals_trg ** (i)))
            new_column_valid = pd.Series(np.array(self.X_vals_valid ** (i)))
            col_str = 'w'+ str(i)
            self.basis_function_trg[col_str] = new_column_trg
            self.basis_function_valid[col_str] = new_column_valid
            i+=1 
        print self.basis_function_trg
        print self.basis_function_valid
        
    def coeffs(self) :
        first = inv(self.basis_function_trg.T.dot(self.basis_function_trg))
        second = self.basis_function_trg.T.dot(self.Y_vals_trg)
        self.betas = first.dot(second)
        print "Betas are: %s" % (self.betas)

    def empirical_error(self):
#        print "Shape of X_Vals_data : %d" % (self.X_vals_data.shape)
        emp_error = self.Y_vals_trg - (self.basis_function_trg.dot(self.betas))
        emp_sq_error = emp_error ** 2
        self.emp_error = emp_sq_error.sum()/emp_sq_error.size
        print "Emperical Error is : %f" % (self.emp_error)
        
    def generalization_error(self):
        print "printing Y_vals_valid"        
        print self.Y_vals_valid
        print "Printing basis function valid"
        print self.basis_function_valid
        print "Betas inside"
        print self.betas
        e = self.Y_vals_valid - (self.basis_function_valid.dot(self.betas))
        sq_error = e ** 2
        self.gen_error = sq_error.sum()/sq_error.size
        print "Generalization error is : %f" % (self.gen_error)
        