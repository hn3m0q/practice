import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeatureScaling(object):   
    def rescaling(self, X):
        '''scale the range in [0, 1]'''
        for col in range(X.shape[1]):
            if X[:, col].ptp() != 0.0:
                X[:, col] = (X[:, col] - X[:, col].min()) / (X[:, col]).ptp()
            else:
                X[:, col] = 0
        return X
    
    def standardization(self, X):
        for col in range(X.shape[1]):
            if X[:, col].std() != 0.0:
                X[:, col] = (X[:, col] - X[:, col].mean()) / X[:, col].std()
            else:
                X[:, col] = 0
        return X

class GradientDescent(object):
    def __init__(self, lr, max_iter, max_errors):
        '''
        lr: learning rate for gradient descent
        max_iter: maximum number of epoches
        max_errors: maximum number of errors that can be tolerated
        '''
        self.lr = lr
        self.max_iter = max_iter
        self.max_errors = max_errors
