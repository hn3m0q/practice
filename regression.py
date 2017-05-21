import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
run feature_scaling => preprocessing => learning model

TODO: what if X[:, col].ptp() = 0?
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
    def __init__(self, lr, max_iter = float('inf'), max_cost = 0):
        '''
        lr: learning rate for gradient descent
        max_iter: maximum number of epoches
        max_cost: maximum cost that can be tolerated
        '''
        self.lr = lr
        self.max_iter = max_iter
        self.max_cost = max_cost
    
    def hypothesis(self, X, theta):
        return np.dot(X, theta)
    
    def batch(self, X, y, theta):
        '''
        here gives two ways to teminate the loop
        choose max_iter or max_cost or both
        '''
        counter = 0 # counter of epoches
        m = X.shape[0] # number of datasets
        temp_cost = float('inf') # assign for the first loop
        while True:
            cost = (1 / (2 * m)) * np.sum((self.hypothesis(X, theta) - y) ** 2)
            
            # break info for choosing better parameter for gradient descent
            if cost <= self.max_cost:
                print('\033[92m' + 'cost threshold reached break' + '\033[00m')
                break
            if counter == self.max_iter:
                print('\033[92m' + 'max iterations reached break' + '\033[00m')
                break
            if cost > temp_cost: 
                print('\033[91m' + 'increasing cost break' + '\033[00m')
                break
            
            #update
            theta -= self.lr * np.dot(X.T, self.hypothesis(X, theta) - y) / m
            temp_cost = cost # convergence watcher
            counter += 1
        return theta

class normal_equation(object):
    def test(self, X):
        if np.linalg.det(np.dot(X.T, X)) == 0.0:
            return False
        else:
            return True
    
    def run(self, X, y):
        if self.test(X):
            return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        else:
            sys.exit(('\033[91m' + 'singular matrix X' + '\033[00m'))

if __name__ == '__main__':
    import random
    
    def gen_data(m, n, varience):
        '''
        m = number of datasets
        n = number of features
        varience / 2 = max range between varied y and raw y
        '''
        
        #generate target theta, the first column is for bias
        theta = 10 * np.random.rand(n + 1, 1)
        print('set theta:\n', theta.T, '\n')
        
        # set the range of x to be 5
        X = 5 * np.random.rand(m, n)
        raw_y = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)
        # y for training
        y = raw_y + varience * np.random.randn(m, 1)
        return X, y
    
    def test(X, y, theta):
        # gradient descent
        print('predicted theta with batch gradient descent:')
        theta = gd.batch(X, y, theta)
        print(theta.T, '\n')
        
        # normal equation
        print('theta with normal equation:')
        theta = nq.run(X, y)
        print(theta.T, '\n')
    
    pp = preprocessing()
    fs = feature_scaling()
    gd = gradient_descent(lr = 0.01, max_iter = 10000)
    nq = normal_equation()
    
    # generate X and y
    X, y = gen_data(100, 5, 1)
    
    # init X and theta, initial theta for gradient descent
    X = pp.X_init(X)
    theta = pp.theta_init(X)
    
    # run test
    test(X, y, theta)