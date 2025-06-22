#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:04:38 2025

@author: xbb
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from o1Neuro.o1Neuro import o1Neuro


class o1NeuroBoost:
    def __init__(self, p, eta = 0.2, M = 10, 
                 K=100, sparsity_level=2, stochastic_ratio = 1.0,
                 stabilizer = 0):
        # M: number of weak learners in boosting
        # b: number of update iterations (for each weak learner)
        self.p = p
        self.eta = eta # learning rate
        self.sparsity_level = sparsity_level
        
        self.K = K

        
        self.X0 = None
        # self.y_mean = None
        self.y0 = None
        self.baggings = None
        self.stochastic_ratio = stochastic_ratio
        self.stabilizer = stabilizer
        self.r2_record = -float('Inf')
        
        # Initialize o1Neuros
        self.o1neuros = [o1Neuro(p, 
                                 K=K, 
                                 sparsity_level=sparsity_level,
                                 stochastic_ratio = stochastic_ratio,
                                 stabilizer = stabilizer
                                 ) for i in range(M)]
        
    def train_reboost(self, X, y, b=100, bootstrap=True, K = None):
        M = len(self.o1neuros)
        self.X0 = X.copy()
        self.y0 = y.copy()
        self.residuals = [np.zeros(self.y0.shape) for _ in range(M)]
        self.residuals[0] = self.y0
        

        

        for _ in range(b):
            for i, network in enumerate(self.o1neuros):
                # Transfter learning:
                W_star, C_star = network.W_star, network.C_star
                # Update residuals of o1Neuros
                network.train(self.X0, 
                              self.residuals[i].copy(), 
                              b = 1,
                              W_star = W_star, 
                              C_star = C_star,
                              K = K)
                
                if i < M - 1:
                    self.residuals[i + 1] = \
                        self.residuals[i] - self.eta * network.predict(self.X0)
                        
    # def train_stable(self, X, y, test_size = 0.2, b = 100):
    
        
    #     X0_temp = X
    #     y0_temp = y
        
    #     # Optimizing based on stable prediction evaluation
    #     # test_size ranges from 0.1 to 0.5
    #     X_train, X_test, y_train, y_test\
    #         = train_test_split(X, y, test_size = test_size) 
    #     networks = self.copy()
    #     for _ in range(b):
            
    #         self.train_reboost(X_train, y_train, b = 1)
            
    #         # Sample indices with replacement
    #         r2_temp = float('Inf')
    #         for _ in range(50):
    #             bootstrap_indices = np.random.choice(len(y_test), 
    #                                                  size=len(y_test), 
    #                                                  replace=True)
    #             X_boot = X_test[bootstrap_indices]
    #             y_boot = y_test[bootstrap_indices]
    #             y_pred = self.predict(X_boot, bagging = False)
    #             r2_temp = min(r2_temp, r2_score(y_boot, y_pred))
    
    #         if r2_temp > self.r2_record:
    #             self.r2_record = r2_temp                
    #             networks = self.copy()
    
    
    #     # Keep the best weights and biases
    #     self.o1neuros = networks
    #     M = len(self.o1neuros)
    #     # Calculate the residuals based on the input (X, y)
    #     residuals = [np.zeros(len(y0_temp)) for _ in range(M)]
    #     residuals[0] = y0_temp
    #     self.o1neuros[0].y0 = residuals[0]
    
    #     for l, network in enumerate(self.o1neuros):
    #         network.X0 = X0_temp.copy()
    #         if l < M - 1:
    #             residuals[l + 1] = \
    #                 residuals[l] - self.eta * network.predict(X0_temp)
    #             networks[l + 1].y0 = residuals[l + 1]
                
    #     self.X0 = X0_temp
    #     # self.y_mean = y_mean_temp
    #     self.y0 = y0_temp
        
    #     return None
    
    def copy(self):
        M = len(self.o1neuros)
        networks = [o1Neuro(self.p, 
                            K=self.K,
                            sparsity_level=self.sparsity_level,
                            stochastic_ratio = self.stochastic_ratio,
                            stabilizer = self.stabilizer
                            ) for i in range(M)]
        for l, network in enumerate(networks):
            # Copy weights and biases
            # Do not copy residuals
            network.W_star = self.o1neuros[l].W_star
            network.C_star = self.o1neuros[l].C_star
        return networks
    
    
    def get_params(self):

        return  {"sparsity_level" : self.sparsity_level, 
                "stochastic_ratio" : self.stochastic_ratio,
                "M" : len(self.o1neuros),
                "p" : self.p,
                'stabilizer' : self.stabilizer}
    
    
    def train_bagging(self, b = 10, n_estimators = 5):
        models = []
        n_samples = self.X0.shape[0]
        for i in range(n_estimators):
            # Sample indices with replacement
            bootstrap_indices = np.random.choice(n_samples, 
                                                 size=n_samples, 
                                                 replace=True)            
            X_boot = self.X0[bootstrap_indices]
            y_boot = self.y0[bootstrap_indices] 
            # Train a weak predictor on the bootstrap sample
            model = o1NeuroBoost(self.p, 
                                 eta = self.eta, 
                                 M = len(self.o1neuros),
                                 K = 10,
                                 sparsity_level=2,
                                 stochastic_ratio = self.stochastic_ratio,
                                 stabilizer = self.stabilizer)
            model.o1neuros = self.copy()            
            model.train_reboost(X_boot, y_boot, b = b)
            models.append(model)
        self.baggings = models
        
    def bagging_predict(self, X_test):
        if self.baggings is None:
            print('Train baggining predictors first.')
            return None
        y_pred = np.zeros(X_test.shape[0])
        for model in self.baggings:
            y_pred += model.predict(X_test, bagging = False)
        y_pred /= len(self.baggings)
        return y_pred
    
    def predict(self, X_test, bagging = True):
        if self.baggings is None:
            bagging = False
        if bagging:
            return self.bagging_predict(X_test)
        if self.y0.ndim > 1:
            y_pred = np.zeros((X_test.shape[0], self.y0.shape[1]))
        else:
            y_pred = np.zeros(X_test.shape[0])
        for i, network in enumerate(self.o1neuros):
            y_pred += self.eta * network.predict(X_test)
        return y_pred