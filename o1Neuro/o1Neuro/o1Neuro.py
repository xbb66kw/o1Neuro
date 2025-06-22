#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 09:31:26 2025

@author: xbb
"""

import random
import numpy as np


from o1Neuro.util.sparsity_matrix import sparsity_matrix_fast



class o1Neuro:
    def __init__(self, p, K=1000, sparsity_level=2, stochastic_ratio = 1.0,
                 stabilizer = 0):
        # b is the number of iterations
        self.K = K + 1
        self.sparsity_level = sparsity_level # w0
        self.p = p
        
        self.W_star = None
        self.C_star = None
        
        self.stochastic_ratio = stochastic_ratio
        self.stabilizer = stabilizer

    def train(self, 
              X,
              y,
              b=1000,
              W_star = None, 
              C_star = None, 
              K = None):
        self.X0 = X.copy()
        self.y0 = y.copy()
        
        # Initialize weights and biases
        p = [self.X0.shape[1]] + self.p
        num_layers = len(p) - 1
        self.W_star = [np.zeros((p[l], p[l+1])) for l in range(num_layers)]
        self.C_star = [np.zeros(p[l+1]) for l in range(num_layers)]
        
        
        self.K = K or self.K
        
        # Transfer learning or not
        self.W_star = W_star if W_star is not None else self.W_star
        self.C_star = C_star if C_star is not None else self.C_star
        for _ in range(b):
            self.sequential_training()

    def sequential_training(self):
        '''
        
    
        Parameters
        ----------

        W_star : list
            W_star[0], ... W_star[L-1], where L = len(W_star) represents the 
            number of hidden layers plus one output layer.
            W_star[l] is a (p[l] by p[l+1]) ndarray, where p[l] is the number 
            of neural nets at the lth layer. The 0th layer is the input layer.
        C_star : list
            DESCRIPTION.
        K : TYPE, optional
            K + 1 is the number of candidate parameter sets for optimization.
            The default is 1000.
        sparsity_level : TYPE, optional
            DESCRIPTION. The default is 2.
    
        '''
        # X_train, y_train, = self.X, self.y
        W_star = [w.copy() for w in self.W_star]
        C_star = [c.copy() for c in self.C_star]
        n = int(self.X0.shape[0] * self.stochastic_ratio)
        p = [self.X0.shape[1]] + self.p
        K, sparsity_level = self.K, self.sparsity_level
        
        ###
        ###
        # TESTED FEATURE
        # CURRENTLY DISABLED. YOU MAY COMMENT THEM OUT
        # Sample the updated neurons in this round
        update_list = []
        for l in range(len(W_star)):
            for h in range(self.p[l]):
                update_list.append((l, h))
        k = len(update_list)
        # Set it to less than 100 to activate this feature
        # retain_count = max(1, int(k * 50 / 100))
        retain_count = max(1, int(k * 100 / 100))
        update_list = random.sample(update_list, retain_count)
        ###
        
        for i_, weight_matrix in enumerate(reversed(W_star)):
            l = len(W_star) - 1 - i_
            n_ = weight_matrix.shape[1]
            for r in range(n_):
                
                
                ####
                ####
                # Stochastic EEO
                ind_set = np.random.choice(
                    a = self.X0.shape[0], 
                    size = n,
                    replace = False)
                X_train, y_train, = self.X0[ind_set, :], self.y0[ind_set]
                ####
                ####
                
                if not (l, r) in update_list:
                    continue
                
                
                
                # Broadcast weights and biases for all layers
                W = [np.broadcast_to(W_star[q][:, :, None], 
                        (p[q], p[q+1], K)) for q in range(len(W_star))]
                C = [np.broadcast_to(C_star[q][:, None], 
                        (p[q+1], K)) for q in range(len(C_star))]
    
                W[l] = W[l].copy()
                C[l] = C[l].copy()
    
                # Check if connected to output
                # The last layer is always connected to output                    
                connected_ = True
                connected_ = l < len(W_star) - 1
                connected_set = []
                column_ = [0]
                for l_ in range(len(W_star) - 1, 0, -1):
                    indices = np.where(np.sum(W_star[l_][:, column_]**2, axis=1) > 0)[0]
                    
                    for elem in indices:
                        connected_set.append((l_-1, elem))
                    column_ = indices
                connected_ = connected_ and (l, r) in connected_set
                ####
                ####


                # Sparsify random weights
                W_temp = np.random.randn(p[l], K)
                W_temp = sparsity_matrix_fast(W_temp, sparsity_level)
                W_temp /= np.linalg.norm(W_temp, axis=0, keepdims=True)                
                    
                W_temp = W_temp[:, np.newaxis, :]                
                W[l][:, r, :][:, None, :] = W_temp
                # The last one is the current weight vectors
                W[l][:, r, K-1] = W_star[l][:, r]
                

                # Generate bias
                if l == 0:
                    random_int_list = np.random.randint(0, n, size=K)
                    M = X_train[random_int_list, :].T.reshape(p[0], 1, K)
                else:
                    M = np.random.randint(0, 2, size=(p[l], 1, K))
                

                C[l][r, :] = np.sum(M * W[l][:, r, :][:, None, :], axis=0)[0, :]
                
                C[l][r, K-1] = C_star[l][r]
                
    
                if connected_:

                    # Forward pass
                    W_out = [np.einsum('ij,jlk->ilk', X_train, W[0]) > C[0]]
                    for t in range(1, len(p) - 1):
                        W_out.append(np.einsum('ijk,jlk->ilk', 
                                               W_out[t-1], 
                                               W[t]) > C[t])
                        
                        
    
                    subsamplesize1 = np.sum(W_out[-1][:, 0, :], axis=0)
                    subsamplesize2 = n - subsamplesize1
                    subsamplesize1[subsamplesize1 == 0] = 1
                    subsamplesize2[subsamplesize2 == 0] = 1
                    
                    # Shape: (y_train.shape[1], K)
                    if len(y_train.shape) > 1:
                        sum1 = np.sum(np.einsum('il,ik->ilk', 
                                                y_train, 
                                                W_out[-1][:, 0, :]), axis = 0)
                        sum2 = np.sum(y_train, axis = 0)[:, None] - sum1
        
                        gain_vector = sum1**2 / subsamplesize1 \
                            + sum2**2 / subsamplesize2
                        gain_vector = np.sum(gain_vector, axis = 0)
                    else:
                        sum1 = y_train @ W_out[-1][:, 0, :]
                        sum2 = np.sum(y_train) - sum1
        
                        gain_vector = sum1**2 / subsamplesize1 \
                            + sum2**2 / subsamplesize2

                    # There are some small numerical differences 
                    # at 13~15 decimal places
                    gain_vector[K-1] += np.var(y_train) * self.stabilizer
                    max_ind = np.argmax(gain_vector)
                else:
                    max_ind = np.random.choice(K)

                W_star = [w_[:, :, max_ind].copy() for w_ in W]
                C_star = [c_[:, max_ind].copy() for c_ in C]

        self.W_star = [w.copy() for w in W_star]
        self.C_star = [c.copy() for c in C_star]

    # For degbugging
    def w_out(self, X):
        # Forward pass
        W = self.W_star
        C = self.C_star
        p = [self.X0.shape[1]] + self.p
        W_out = [np.einsum('ij,jl->il', X, W[0]) > C[0]]
        for t in range(1, len(p) - 1):
            W_out.append(np.einsum('ij,jl->il', W_out[t-1], W[t]) > C[t])
        return W_out[-1]
    def predict(self, X_test):
        X_train, y_train = self.X0, self.y0
        W_star, C_star = self.W_star, self.C_star

        
        # fitting the model
        W_temp = X_train
        for l in range(len(W_star)):
            W_temp = W_temp @ W_star[l]
            W_temp = W_temp > C_star[l]

        

        arr = y_train[W_temp[:, 0]]
        one_prediction = arr.mean(axis = 0) if arr.size > 0 else 0.0
        arr = y_train[~W_temp[:, 0]]
        zero_prediction =  arr.mean(axis = 0) if arr.size > 0 else 0.0
        
        # making prediction
        W_temp = X_test
        for l in range(len(W_star)):
            W_temp = W_temp @ W_star[l]
            W_temp = W_temp > C_star[l]

        y_pred = one_prediction * W_temp

        W_temp = W_temp[:, 0] if W_temp.ndim > 1 else W_temp
        y_pred[~W_temp] = zero_prediction

        if len(y_train.shape) == 1:
            return y_pred[:, 0]
        else:
            return y_pred
