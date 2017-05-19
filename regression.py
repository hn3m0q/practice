import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
run feature_scaling => preprocessing => learning model

TODO: what if X[:, col].ptp() = 0?
TODO: underfitting and overfitting?
'''

class preprocessing(object):
    def X_init(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    def theta_init(self, theta):
        # run X_init before theta_init
        return np.zeros((X.shape[1], 1))

class feature_scaling(object):
    def rescaling(self, X):
        '''scale the range in [0, 1]'''
        for col in range(X.shape[1]):
            if X[:, col].ptp() != 0.0:
                X[:, col] = (X[:, col] - X[:, col].min()) / X[:, col].ptp()
        return X
    
    def standardization(self, X):
        for col in range(X.shape[1]):
            if X[:, col].std() != 0.0:
                X[:, col] = (X[:, col] - X[:, col].mean()) / X[:, col].std()
        return X

class gradient_descent(object):
    def __init__(self, lr, max_iter, max_errors = 0):
        '''
        lr: learning rate for gradient descent
        max_iter: maximum number of epoches
        max_errors: maximum number of errors that can be tolerated
        '''
        self.lr = lr
        self.max_iter = max_iter
        self.max_errors = max_errors
    
    def hypothesis(self, X, theta):
        return np.dot(X, theta)
    
    def batch(self, X, y, theta):
        '''
        here gives two ways to teminate the loop
        choose max_iter or max_errors or both
        '''
        c = 0 # counter of epoches
        m = X.shape[0] # number of datasets
        temp_cost = float('inf') # assign for the first loop
        while True:
            cost = (1 / (2 * m)) * np.sum((self.hypothesis(X, theta) - y) ** 2)
            if cost <= self.max_errors or cost > temp_cost: break
            if c == self.max_iter or cost > temp_cost: break
            
            theta -= self.lr * np.dot(X.T, self.hypothesis(X, theta) - y) / m
            temp_cost = cost # convergence watcher
            c += 1
        return theta, cost

#test
if __name__ == '__main__':
    import random
    
    # generate target theta, the first column is for bias
    target_theta = 10 * np.random.rand(5, 1)
    # set varience
    varience = 1
    # generate data
    X = np.zeros((20, 4))
    y = np.zeros((20, 1))
    for row in range(20):
        for col in range(4):
            X[row, col] = row + varience * random.random()
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = np.dot(X, target_theta)
    
    print('target_weights:\n', target_theta)
    
    pp = preprocessing()
    fs = feature_scaling()
    gd = gradient_descent(lr = 0.001, max_iter = 1000)
    
    X = pp.X_init(fs.standardization(X))
    theta = pp.theta_init(X)
    prediction = gd.batch(X, y, theta)[0]
    final_cost = gd.batch(X, y, theta)[1]
    
    print('prediction theta:\n', prediction)
    print('final cost:\n', final_cost)
