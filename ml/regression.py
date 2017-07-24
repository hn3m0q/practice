import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
run feature_scaling => preprocessing => learning model

TODO: what if X[:, col].ptp() = 0?
TODO: SGD parameters
'''

class preprocessing(object):
    def X_init(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    def theta_init(self, X):
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
    
    def hyp(self, X, theta):
        ''' hypothesis function '''
        return np.dot(X, theta)

    def mse_cost(self, X, y, theta):
        m = X.shape[0] # number of datasets
        return (1 / (2 * m)) * np.sum((self.hyp(X, theta) - y) ** 2)
    
    def batch(self, X, y, theta):
        '''
        here gives two ways to teminate the loop
        choose max_iter or max_cost or both
        '''
        counter = 0 # counter of epoches
        m = X.shape[0] # number of datasets
        temp_cost = float('inf') # assign for the first loop

        while True:
            cost = self.mse_cost(X, y, theta)
            
            # break info for choosing better parameter for gradient descent
            if cost <= self.max_cost:
                print('\033[92m' + 'cost threshold reached break' + '\033[00m')
                print('iteration:', counter)
                break
            if counter == self.max_iter:
                print('\033[92m' + 'max iterations reached break' + '\033[00m')
                print('iteration:', counter)
                break
            if cost > temp_cost:
                print('\033[91m' + 'increasing cost break' + '\033[00m')
                print('iteration:', counter)
                break
            
            theta -= self.lr * np.dot(X.T, self.hyp(X, theta) - y) / m
            temp_cost = cost # convergence watcher
            counter += 1
        return theta
    
    def stochastic(self, X, y, theta, t0 = 1, t1 = 200):
        counter = 0 # counter of epoches
        m = X.shape[0] # number of datasets
        
        learning_rate = lambda t : t0 / (t + t1)
        
        while True:
            cost = self.mse_cost(X, y, theta)

            # break info for choosing better parameter for gradient descent
            if cost <= self.max_cost:
                print('\033[92m' + 'cost threshold reached break' + '\033[00m')
                print('iteration:', counter)
                break
            if counter == self.max_iter:
                print('\033[92m' + 'max iterations reached break' + '\033[00m')
                print('iteration:', counter)
                break

            for row in range(m):
                index = np.random.randint(m)
                xi = X[index:index + 1]
                yi = y[index]
                lr = learning_rate(counter * m + row)
                theta -= lr * 2 * xi.T * (self.hyp(xi, theta) - yi)

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
    def gen_data(m, n, varience):
        import random
        '''
        m = number of datasets
        n = number of features
        varience / 2 = max range between varied y and raw y
        '''
        
        #generate target theta, the first column is for bias
        theta = 10 * np.random.rand(n + 1, 1)
        print('\033[95m' + 'set theta:' + '\033[00m')
        print(theta.T, '\n')
        
        # set the range of x to be 5
        X = 5 * np.random.rand(m, n)
        raw_y = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)
        # adding varience for training
        y = raw_y + varience * np.random.randn(m, 1)
        return X, y
    
    def copy():
        # prevent list address issue
        global X_test
        global y_test
        global theta_test
    
        X_test = np.copy(X)
        y_test = np.copy(y)
        theta_test = np.copy(theta)       
    
    def test(X, y, theta):
        # use codes above to solve regression
        fs = feature_scaling()
        gd = gradient_descent(lr = 0.01, max_cost = 0.4)
        nq = normal_equation()
        
        # gradient descent
        copy()
        print('\033[94m' + 'batch gradient descent:' + '\033[00m')
        theta_BGD = gd.batch(X_test, y_test, theta_test)
        print(theta_BGD.T, '\n')
        
        copy()
        print('\033[94m' + 'stochastic gradient descent:' + '\033[00m')
        theta_SGD = gd.stochastic(X_test, y_test, theta_test)
        print(theta_SGD.T, '\n')
        
        # normal equation
        copy()
        print('\033[94m' + 'normal equation:' + '\033[00m')
        theta_NQ = nq.run(X_test, y_test)
        print(theta_NQ.T, '\n')
    
    def verify(X, y, theta):
        # use machine learning libraries to sovle regression
        
        from sklearn.linear_model import LinearRegression
        linreg = LinearRegression()
        linreg.fit(X, y)
        linreg.coef_[0][0] = linreg.intercept_
        print('\033[93m' + 'sklearn output:' + '\033[00m')
        print(linreg.coef_)
    
    # generate X and y
    X, y = gen_data(100, 5, 1)
    
    # init X and theta, initial theta for gradient descent
    pp = preprocessing()
    X = pp.X_init(X)
    theta = pp.theta_init(X)
    
    # run test
    test(X, y, theta)
    verify(X, y, theta)