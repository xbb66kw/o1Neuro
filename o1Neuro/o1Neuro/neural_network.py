#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:24:07 2025

@author: xbb
"""

import numpy as np
import time

# import torch
# conda
# pip3 install torch torchvision torchaudio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.spatial import distance_matrix


from o1NeuroBoost import o1NeuroBoost
from bnn.tree_functions import rf_train, xgb_train, o1neuroboost_train


# import cupy as cp
#%%



p = [100, 8, 4, 1] # the last layer is 1
# p = [100, 16, 8, 4, 1] # the last layer is 1


#%%

# Generate a sample of size n
n = 200
X = np.random.uniform(-2*np.pi, 2*np.pi, size=(n, p[0]))
print(f'X.shape : {X.shape}')
type_ = 1 # [1,2,3]
if type_ == 1:
    # Polynomial Regression
    y = 2 * X[:, 0]**2 - 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n)
elif type_ == 2:
    # Sinusoidal Interaction
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(n)
elif type_ == 3:
    # Multiplicative Interaction
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n)


#%%


n_test = 10000
X_test = np.random.uniform(-2*np.pi, 2*np.pi, size=(n_test, p[0]))

if type_ == 1:
    # Polynomial Regression
    y_test = 2 * X_test[:, 0]**2 - 3 * X_test[:, 1] + 0.5 * X_test[:, 2]
elif type_ == 2:
    # Sinusoidal Addition
    y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])
elif type_ == 3:
    # Multiplicative Interaction
    y_test = X_test[:, 0] * X_test[:, 1] + 0.5 * X_test[:, 2]


#%%

# Train boosted 01Neuro 
eta = 0.2 # 0.2, 0.25, 0.30, 0.35, 0.4
M = 20 # 4, 5, 6
# Regular tabular regression tests:
print(f'Architecture (p[0] is the number of original features): {p}, ', \
      f'eta : {eta}, M : {M}')
snb = o1NeuroBoost(p[1:], eta = eta, M = M,
                   K = 10, sparsity_level=2)

start_time = time.time()
# Default do not tune:
# bootstrap=True, transfer = True
snb.train_stable(X, y, b=40)

end_time = time.time()

print(f'used time : {start_time - end_time}')

y_pred = snb.predict(X)
r2 = r2_score(y, y_pred)
print(f'r2 training : {r2}')

y_pred = snb.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'r2 : {r2}')

#%%


snb = o1NeuroBoost([8, 4, 1], eta = 0.2, M = 15,
                   K = 10, sparsity_level=2)


snb.train_stable(X, y, b= 5)



y_pred = snb.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'r2 : {r2}')


# #%%

# snb.bagging(b = 1, n_estimators = 5)
# y_pred = snb.bagging_predict(X_test)
# r2 = r2_score(y_test, y_pred)
# print(f'r2 : {r2}')
# #%%
# # Continue training
# snb.train_reboost(X, y, b=100)



# network = snb.sparseneuros[0]
# # network.train(b=100, K = 100)

# r2_score(y, eta * network.predict(X) + np.mean(y))



# y_pred = snb.predict(X)
# r2 = r2_score(y, y_pred)
# print(f'r2 training : {r2}')

# y_pred = snb.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# print(f'r2 : {r2}')


# #%%
# eta = 1
# k_star = 0
# k_ = 5
# binary_v = np.ones((len(y), k_ + 1))
# for l in range(k_star, k_):
#     binary_v[:, l + 1] = snb.sparseneuros[l].w_out(X)[:, 0]
 
# beta, residuals, rank, s = np.linalg.lstsq(binary_v, y, rcond=None)




# # Training R-squared
# train_rss = np.sum((y - binary_v @ beta * eta) ** 2)
# train_tss = np.var(y) * len(y)
# r2_train = 1 - train_rss / train_tss
# print(f'r2_train : {r2_train}')


# binary_v = np.ones((len(y_test), k_ + 1))
# for l in range(k_star, k_):
#     binary_v[:, l + 1] = snb.sparseneuros[l].w_out(X_test)[:, 0]
 
# # Training R-squared
# test_rss = np.sum((y_test - binary_v @ beta * eta) ** 2)
# test_tss = np.var(y_test) * len(y_test)
# r2_test = 1 - test_rss / test_tss
# print(f'r2_test : {r2_test}')


# #%%
# # Assign a 2D physical position (e.g., random coordinates in a square)
# positions = np.random.rand(p[0], 2)  # shape: 20 × 2

# kappa = 5  # for example

# # Compute pairwise Euclidean distance matrix
# dists = distance_matrix(positions, positions)

# # Get the indices of the kappa closest neighbors for each variable
# # axis=1 means row-wise sort; np.argsort returns indices of sorted array
# neighbor_matrix = np.argsort(dists, axis=1)[:, :kappa]

# #%%

# # Generate a sample of size n
# n = 200
# # number of averaging featrues for each factor
# s0 = 3
# # common factors
# n_factor = 3

# X = np.random.uniform(-2*np.pi, 2*np.pi, size=(n, p[0]))

# Z_train = np.zeros((n, n_factor))
# rn_ind_sets = []
# for index in range(n_factor):
#     a = neighbor_matrix[index, :]
#     rn_ind_sets.append(np.random.choice(a, size = s0, replace=False))
#     rn_ind_set = rn_ind_sets[index]
#     Z_train[:, index] = np.mean(X[:, rn_ind_set], axis=1)



# print(f'X.shape : {X.shape}')
# type_ = 3 # [1,2,3,4]
# if type_ == 1:
#     # Polynomial Regression
#     y = 2 * Z_train[:, 0]**2 - 3 * Z_train[:, 1] + 0.5 * Z_train[:, 2] + np.random.randn(n)
# elif type_ == 2:
#     # Sinusoidal Interaction
#     y = np.sin(Z_train[:, 0]) + np.cos(Z_train[:, 1]) + 0.1 * np.random.randn(n)
# elif type_ == 3:
#     # Multiplicative Interaction
#     y = Z_train[:, 0] * Z_train[:, 1] + 0.5 * Z_train[:, 2] + np.random.randn(n)
# elif type_ == 4:
#     # Piecewise Function
#     y = np.where(Z_train[:, 0] + Z_train[:, 1] > 0, 1, -1) + 0.2 * np.random.randn(n)


# #%%


# n_test = 10000
# X_test = np.random.uniform(-2*np.pi, 2*np.pi, size=(n_test, p[0]))

# Z_test = np.zeros((n_test, n_factor))
# for index in range(n_factor):
#     rn_ind_set = rn_ind_sets[index]
    
#     Z_test[:, index] = np.mean(X_test[:, rn_ind_set], axis=1)




# if type_ == 1:
#     # Polynomial Regression
#     y_test = 2 * Z_test[:, 0]**2 - 3 * Z_test[:, 1] + 0.5 * Z_test[:, 2]
# elif type_ == 2:
#     # Sinusoidal Interaction
#     y_test = np.sin(Z_test[:, 0]) + np.cos(Z_test[:, 1])
# elif type_ == 3:
#     # Multiplicative Interaction
#     y_test = Z_test[:, 0] * Z_test[:, 1] + 0.5 * Z_test[:, 2]
# elif type_ == 4:
#     # Piecewise Function
#     y_test = np.where(Z_test[:, 0] + Z_test[:, 1] > 0, 1, -1)


# #%%

# # Hyperparameters:
# # K + 1 is the number of parameter sets for optimization
# # M is the number of boosted weak predictors
# # eta is the boost learning rate
# # s0 is the number of averaging predictors
# # sparsity_level is w0

# # If X_neighbor = neighbor_matrix is provided, spatial 01Neuro is used.

# rn_ind_sets[0]
# rn_ind_sets[1]
# l = 6
# snb.sparseneuros[0].W_star[0][:, l][rn_ind_sets[0]]
# snb.sparseneuros[0].W_star[0][:, l][rn_ind_sets[1]]

# snb = o1NeuroBoost(p[1:], eta = 0.3, M = 1,
#                    K = 50, sparsity_level=2,
#                    s0 = 3, X_neighbor = neighbor_matrix)

# snb.train(X, y, b=100)

# y_pred = snb.predict(X_test)
# r2 = r2_score(y_test, y_pred)

# print(f'r2: {r2}')

# #%%

# for _ in range(1000):
#     # Continue training
#     snb.train_reboost(b=100)
    
#     y_pred = snb.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
    
#     print(f'r2: {r2}')


# #%%

# # Initialize Random Forest Regressor
# rf = RandomForestRegressor(n_estimators=100, max_depth=None, max_features = 0.9)

# # Train the model
# rf.fit(X, y)

# # Predict on test set
# y_pred = rf.predict(X_test)

# # Evaluation metrics
# r2 = r2_score(y_test, y_pred)
# r2
