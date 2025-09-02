#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 09:31:26 2025

@author: xbb
"""

import random
import numpy as np
from numba import njit, prange


@njit(cache=True)
def compute_residuals_standalone(W_temp_col, y_train, eta):
    n_samples = len(y_train)
    count_1 = 0.0
    for i in range(n_samples):
        if W_temp_col[i] > 0.5:
            count_1 += 1.0
    
    count_0 = float(n_samples) - count_1
    count_0 = max(count_0, 1.0)
    count_1 = max(count_1, 1.0)
    
    sum_0 = sum_1 = 0.0
    for i in range(n_samples):
        if W_temp_col[i] > 0.5:
            sum_1 += y_train[i]
        else:
            sum_0 += y_train[i]
    
    ave_0 = sum_0 / count_0
    ave_1 = sum_1 / count_1
    
    residual = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if W_temp_col[i] > 0.5:
            residual[i] = y_train[i] - ave_1 * eta
        else:
            residual[i] = y_train[i] - ave_0 * eta
    
    return residual

class o1Neuro_jit:
    
    
    def __init__(self, 
                 p = [900],
                 eta = 0.1, 
                 K=30, # 60
                 sparsity_level=2, 
                 stochastic_ratio_init = 0.05, # 0.1
                 stochastic_ratio = 1.0, # 1.0
                 stabilizer = 0.01,
                 tuning_phase = False,
                 params = None):
        if params is not None:
            self.K = params['K']
            self.eta = params['eta']
            self.p = params['architecture']
        else:
            if p is None:
                raise ValueError("p is a required parameter for initialization")
            # b is the number of iterations
            self.K = K
            self.p = [int(n_neurons) for n_neurons in p]
            
            
            self.eta = eta
            
        self.W_star = None
        self.C_star = None
        self.sparsity_level = sparsity_level # w0
        self.stochastic_ratio_init = stochastic_ratio_init
        self.stochastic_ratio = stochastic_ratio
        self.stabilizer = stabilizer
        self.tuning_phase = tuning_phase
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def einsum_2d_jkl(W_out_temp, W):
        n, j = W_out_temp.shape
        j2, k, l = W.shape
        out = np.empty((n, k, l), dtype=np.float64)
        for i in prange(n):
            for kk in range(k):
                for ll in range(l):
                    s = 0.0
                    for jj in range(j):
                        s += W_out_temp[i, jj] * W[jj, kk, ll]
                    out[i, kk, ll] = s
        return out
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def einsum_3d_jkl(W_out_temp, W):
        i, j, l = W_out_temp.shape
        j2, k, l2 = W.shape
        out = np.empty((i, k, l), dtype=np.float64)
        for ii in prange(i):
            for kk in range(k):
                for ll in range(l):
                    s = 0.0
                    for jj in range(j):
                        s += W_out_temp[ii, jj, ll] * W[jj, kk, ll]
                    out[ii, kk, ll] = s
        return out
    def train(self, 
              X,
              y,
              b=10,
              W_star = None, 
              C_star = None, 
              K = None):
        self.X0 = X.copy()
        self.y0 = y.copy()

        # Initialize weights and biases
        p = [self.X0.shape[1]] + self.p
        
    
        self.K = K or self.K

        # Transfer learning or not
        if self.W_star is None:
            num_layers = len(p) - 1
            self.W_star = W_star if W_star is not None else \
                [np.zeros((p[l], p[l+1])) for l in range(num_layers)]
            self.C_star = C_star if C_star is not None else \
                [np.zeros(p[l+1]) for l in range(num_layers)]
            
        
        all_neuron_list = []
        for l in range(len(self.W_star)):
            for h in range(self.p[l]):
                all_neuron_list.append((l, h))
        k = len(all_neuron_list)



        # We set self.p_L = 1
        self.M = self.W_star[-1].shape[1]
        for b_ in range(b):
            ###
            ###
            # TRAINING FEATUREs
            n_samples = self.X0.shape[0]
            if self.tuning_phase:
                sample_size = n_samples if n_samples < 300 \
                    else max(300, int(n_samples * self.stochastic_ratio_init))
                self.ind_set = np.random.choice(
                        a = self.X0.shape[0],
                        size = sample_size,
                        replace = False)
                dropout_count = max(1, int(k * 0.6))
            else:                
                sample_size = n_samples if n_samples < 300 \
                    else max(300, int(n_samples * self.stochastic_ratio))
                self.ind_set = np.random.choice(
                        a = self.X0.shape[0],
                        size = sample_size,
                        replace = False)
                dropout_count = 0 # max(1, int(k * 0.4))
            
            # ###
            # # Dropout
            self.dropout_list = random.sample(all_neuron_list, dropout_count)
            self.residuals = [self.y0.copy()]
            
            # From len(self.W_star) - 1 to 0
            # Layer index
            for l in range(len(self.W_star) - 1, -1, -1):
                self.layer_training(l)
               

        
    def update_greedy_lock(self, ind_layer):
        # `ind_layer` is the index of the current layer to be updated
        
        # Create same structure with zeros
        # Greedy locks show the corresponding output predictor for each neuron
        self.greedy_locks = \
            [[(-1, -1) for _ in sublist] for sublist in self.C_star]
            
        # Totally M elemnts, each has depth (L - 1) - ind_layer + 1.
        self.greedy_weight_pyramaid = [[] for _ in range(len(self.C_star[-1]))]        
        self.greedy_bias_pyramaid = [[] for _ in range(len(self.C_star[-1]))]



        for h in range(len(self.C_star[-1])):
            column_ = [h]
            for l in range(len(self.W_star) - 1, ind_layer - 1, -1):
                # Shape (p[l-1], p[l])
                for order_, col in enumerate(column_):
                    
                    if self.greedy_locks[l][col][0] == -1:
                        # order_ is used for indexing later
                        self.greedy_locks[l][col] = (h, order_)

                weight_matrix = self.W_star[l][:, column_]
                bias_vector = np.atleast_1d(self.C_star[l][column_])
                

                if l > ind_layer:
                    # column_ at the next round is the curront rounds's rows
                    column_ = np.where(np.sum(weight_matrix**2, axis=1) > 0)[0]
                    weight_matrix = weight_matrix[column_, :]

                self.greedy_weight_pyramaid[h].append(weight_matrix)
                self.greedy_bias_pyramaid[h].append(bias_vector)


    
    def make_prediction(self, ind_layer, limit=-1):
        """
        Optimized version of make_prediction with several performance improvements:
        1. Specialized for p_L=1 case (binary predictions)
        2. Uses Numba JIT compilation for inner loops
        3. Avoids creating large partition matrices
        4. Pre-allocates arrays
        5. Uses in-place operations where possible
        """
        y_train = self.y0[self.ind_set]
        W_out = self.W_out[ind_layer]
        W_star, C_star = self.W_star, self.C_star
        
        # Forward pass - keep as is since it's already efficient
        W_temp = W_out
        for l in range(ind_layer, len(W_star)):
            W_temp = W_temp @ W_star[l]
            W_temp = W_temp > C_star[l]
        
        # Convert to float for numba compatibility
        W_temp = W_temp.astype(np.float64)
        
        # Initialize residuals
        self.residuals = [y_train.copy()]
        
        if limit == -1:
            limit = self.M
        
        
        # Pre-allocate for reuse
        current_y = y_train.copy()
        
        # Main prediction loop - optimized for p_L=1
        for s in range(min(self.M, limit)):
            # Get current column (binary predictions for this stage)
            W_temp_col = W_temp[:, s]
            
            # Use numba-optimized function
            residual = compute_residuals_standalone(W_temp_col, 
                                         current_y, 
                                         self.eta)
            
            self.residuals.append(residual)
            current_y = residual
            

    def layer_training(self, ind_layer):
        ind_set = self.ind_set
        n = len(ind_set)
        K, sparsity_level = self.K, self.sparsity_level
        X_train = self.X0[self.ind_set, :]
        

        # Update the outputs at each layer
        self.w_out()
        
        # The first layer in self.W_out represents the original input features.
        # This implies that each subsequent W_out corresponds to the output 
        # from the previous layer, serving as the input to the current layer 
        # being updated.
        W_out = self.W_out[ind_layer]
        
        # When updating the current layer (index ind_layer),
        # We only need weights and biases starting from 
        # the current layer.
        self.update_greedy_lock(ind_layer)
        
        # Number of all neurons at the current layer to be updated
        H = self.W_star[ind_layer].shape[1]
        
        # List of all weight vectors at the current layer
        n_all_w = self.W_star[ind_layer].shape[1]
        list_w_all = self.W_star[ind_layer]

        
        # Starts from the lowest cor_output_h
        order_h = []
        for order_ in range(len(self.C_star[-1])):
            for h in range(H):
                # `self.greedy_locks` record each neuron's corresponding 
                # output predictor's index and the column index `r` of this 
                # neuron
                cor_output_h, r = self.greedy_locks[ind_layer][h]
                if cor_output_h == order_:
                    order_h.append(h)
        # Record the complement list of order_h
        order_h_complement = [i for i in range(H) if i not in order_h]
        
        cor_output_h_ = -1
        for h in order_h:
            
            # Dropout
            if (ind_layer, h) in self.dropout_list:
                continue
            
            W_out_temp = W_out
            cor_output_h, r =  self.greedy_locks[ind_layer][h]
            
            if cor_output_h_ != cor_output_h:
                cor_output_h_ = cor_output_h

                # Update boosting residuals `self.residuals`
                self.make_prediction(ind_layer, cor_output_h)
                
            
            
            # Prepare the response
            y_train = self.residuals[cor_output_h]
            
            ind = -1
            for l in range(ind_layer, len(self.W_star)):                
                weight_matrix = self.greedy_weight_pyramaid[cor_output_h][ind]
                bias_vector = self.greedy_bias_pyramaid[cor_output_h][ind]
                ind -= 1

                # Broadcast weights and biases for all layers
                p1, p2 = weight_matrix.shape
                W = np.broadcast_to(weight_matrix[:, :, None], (p1, p2, K))
                C = np.broadcast_to(bias_vector[:, None], (p2, K))    
                W = W.copy()
                C = C.copy()
    
                # Generating K candidatae at ind_layer-th layer
                if l == ind_layer:
                    # Sparsify random weights
                    W_temp = np.random.randn(p1, K)
                    # randomly sample the sparsity level
                    sparsity_ = random.sample(range(sparsity_level), 1)[0] + 1
                    # sparsity_ = 2
                    W_temp = self.sparsity_matrix_fast(W_temp, sparsity_)
                    
                    
                    
                    W_temp /= np.linalg.norm(W_temp, axis=0, keepdims=True)                
                    
                    
                    # Prepare all the candidate weight vectors
                    W[:, r, :] = W_temp
                    
                    # Allocate 50% of candidates from the existing companions
                    q_ = int(K/2)
                    if q_ > 0 and K > 1:
                        column_ind = np.random.choice(np.arange(n_all_w), 
                                                      size=q_, 
                                                      replace=True)
                        W[:, r, :q_] = list_w_all[:, column_ind]

                    ####
                    # The last one is the current weight vectors
                    W[:, r, K-1] = weight_matrix[:, r]
                    
                    
                    ind_set = self.ind_set
                    n = len(ind_set)
                    X_train = self.X0[ind_set, :]
                    # Generate bias
                    if l == 0:
                        random_int_list = np.random.randint(0, n, size=K)
                        M = X_train[random_int_list, :].T.reshape(p1, 1, K)
                    else:
                        M = np.random.randint(0, 2, size=(p1, 1, K))
                    
    
                    C[r, :] = \
                        np.sum(M * W[:, r, :][:, None, :], axis=0)[0, :]
                    # The last one is the current bias
                    C[r, K-1] = bias_vector[r]
                    
                    W_temp = W.copy()
                    C_temp = C.copy()
                
                if len(W_out_temp.shape) == 2:
                    out = self.einsum_2d_jkl(W_out_temp, W)
                else:
                    out = self.einsum_3d_jkl(W_out_temp, W)
                

                W_out_temp = out > C  # broadcastable!

            
            
            # Calculate gain_vector
            # W_out_temp : Shape (len(self.ind_set), K)
            W_out_temp_2 = 1 - W_out_temp

            n_ = np.sum(W_out_temp, axis = 0).reshape(-1).astype(float)
            n_2 = len(y_train) - n_
            # Avoid division by zero
            n_ += 1e-5
            n_2 += 1e-5
            
            
            
            # Sum over samples, each weighted by y_train
            sum_ = np.tensordot(y_train, W_out_temp, axes=(0, 0)).reshape(-1)  
            sum_2 = np.tensordot(y_train, W_out_temp_2, axes=(0, 0)).reshape(-1)
            
            # gain_vector length K
            gain_vector = sum_** 2 / n_ + sum_2 ** 2 / n_2
            

            # There are some small numerical differences 
            # at 13~15 decimal places
            gain_vector[K-1] += gain_vector[K-1] * self.stabilizer            
            max_ind = np.argmax(gain_vector)


            # Update
            self.greedy_weight_pyramaid[cor_output_h][-1][:, r] \
                = W_temp[:, r, max_ind]
            self.greedy_bias_pyramaid[cor_output_h][-1][r] \
                = C_temp[r, max_ind]
            
            # Update
            self.W_star[ind_layer][:, h] = W_temp[:, r, max_ind]
            self.C_star[ind_layer][h] = C_temp[r, max_ind]
        
        # Random update idle neurons
        for h in order_h_complement:
            # Sparsify random weights
            p1 = len(self.W_star[ind_layer][:, h])
            W_temp = np.random.randn(p1, 1)
            sparsity_ = random.sample(range(sparsity_level), 1)[0] + 1
            W_temp = self.sparsity_matrix_fast(W_temp, sparsity_)
            W_temp /= np.linalg.norm(W_temp, axis=0, keepdims=True)
            self.W_star[ind_layer][:, h] = W_temp[:, 0]
            
            
            ind_set = self.ind_set
            n = len(ind_set)
            X_train = self.X0[ind_set, :]
            # Generate bias
            if ind_layer == 0:
                random_int_list = np.random.randint(0, n)
                M = X_train[random_int_list, :]
            else:
                M = np.random.randint(0, 2, size=(p1, 1))
            
            self.C_star[ind_layer][h] = W_temp[:, 0] @ M

    def sparsity_matrix_fast(self, X, s):
        """
        Vectorized version: Retain only `s` randomly chosen elements along axis=0
        for each K position in a (p0, K) matrix.
        """
        p0, K = X.shape
        if p0 > s:
            rand = np.random.rand(p0, K)
            top_s_idx = np.argpartition(rand, s, axis=0)[:s]  # shape: (s, K)



            # Initialize a mask of all True
            mask = np.ones_like(rand, dtype=bool)
        
            # Generate index grids for K to match shape (s, K)
            i2 = np.broadcast_to(np.arange(K)[None, :], (s, K))
            
            # Now set False values in one go
            mask[top_s_idx, i2] = False
            
            # Create sparse mask
            X[mask] = 0
        
        return X # Apply mask

    
    # Output transformed features
    def w_out(self, X = None, layer = None):
        W_temp = X
        if W_temp is None:
            W_temp = self.X0[self.ind_set, :]
        if layer is None:
            layer = len(self.W_star)

        # Forward pass
        W_star, C_star = self.W_star, self.C_star
        W_out = [W_temp]
        for l in range(layer):
            W_temp = W_temp @ W_star[l]
            W_temp = W_temp > C_star[l]
            W_out.append(W_temp)
        self.W_out = W_out
        return W_temp
    
    def get_params(self):
        return {'eta' : self.eta,
                'architecture': self.p,
                'K': self.K}
    
    
    
    def predict(self, X_test, limit = float('Inf')):

        X_train = self.X0
        W_star, C_star = self.W_star, self.C_star

        
        # fitting the model
        W_temp = X_train
        for l in range(len(W_star)):
            W_temp = W_temp @ W_star[l]
            W_temp = W_temp > C_star[l]
        # making prediction
        W_temp_test = X_test
        for l in range(len(W_star)):
            W_temp_test = W_temp_test @ W_star[l]
            W_temp_test = W_temp_test > C_star[l]

        
        y_pred_output = np.zeros(X_test.shape[0])
        
        # W_temp Shape (n_train, p_L * M)
        self.residuals = [self.y0]
        # Currently not aware the benefit of p_L > 1
        p_L = 1
        for s in range(min(self.M, limit + 1)):
            y_train = self.residuals[s]
            # Step 1: binary weights, shape (p,)
            weights = 1 << np.arange(p_L)[::-1]  # e.g., [4, 2, 1] for p=3
            
            # At the output layer, colums s * p_L: (s + 1) * p_L represent
            # those at the s-th boosted predictor
            binary_matrix = W_temp[:, s * p_L: (s + 1) * p_L]
            group_indices = binary_matrix @ weights  # shape (n,)
            partition_matrix = np.eye(2**p_L)[group_indices]  # shape (n, 2^p)
            
            
            nume_ = y_train @ partition_matrix
            deno_ = np.sum(partition_matrix, axis = 0)
            ave_array = np.divide(nume_, 
                                  deno_, 
                                  out=np.zeros_like(nume_), 
                                  where=deno_ != 0)

            
            
            y_pred_train = partition_matrix @ ave_array
            self.residuals.append(y_train - y_pred_train * self.eta)
            
            # For the test data
            binary_matrix_test = W_temp_test[:, s * p_L: (s + 1) * p_L]
            group_indices_test = binary_matrix_test @ weights  # shape (n,)
            partition_matrix_test = np.eye(2**p_L)[group_indices_test]  
            y_pred_test = partition_matrix_test @ ave_array
            
            
            y_pred_output += y_pred_test * self.eta
        return y_pred_output
        

        
