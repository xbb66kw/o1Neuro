#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:03:04 2025

@author: xbb
"""

import time
import warnings, os, pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



from o1Neuro.tree_functions import rf_train, \
    xgb_train, o1Neuro_train, tabnet_train

# Install this package first before usage.
# cd [...]/o1Neuro_github
# pip install -e .

#%%

# Parameters
n = 100  # sample size 100, 500, 3000, 20000
p = 20   # dimensionality
n_test = 10000
n_relevant = 10
weights = np.full(n_relevant, 2.0)

type_ = 0 # 0, 2
#%%



print(f'p : {p}, n_relevant : {n_relevant}, n : {n}')


#%%
B = 20
run_tabnet = True
run_cat =  True
run_xgb = True
run_rf = True

save_ = False

#%%


def _rng(random_state):
    return np.random.default_rng(random_state)



def additive_model(weights,
                   n_samples=500, 
                   n_features=20, 
                   n_relevant=10,
                   noise=1.0, 
                   make_xor = False, 
                   random_state=None):
    """
    Generate synthetic data from an additive linear model with irrelevant features.
    
    Model: y = sum_{j=1..n_relevant} w_j * x_j + e
    where x_j ~ U(0,1), w_j ~ U(0.5, 2.0), and e ~ N(0, noise^2).
    
    Args:
        n_samples (int): number of samples
        n_features (int): total number of features
        n_relevant (int): number of features that influence y
        noise (float): standard deviation of Gaussian noise
        random_state (int): random seed
    
    Returns:
        X: array (n_samples, n_features)
        y: array (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    
    
    

    # features
    X = rng.uniform(-0.5, 0.5, size=(n_samples, n_features))

    # linear additive contribution
    y = X[:, :n_relevant] @ weights
    
    if make_xor:
        
        U = rng.uniform(-0.5, 0.5, size=(n_samples, n_relevant))
        # 2 * np.random.randint(0, 2, size=X[:, :n_relevant].shape) - 1
        X_scaled = X[:, :n_relevant] * U

        
        y = X_scaled @ weights
        X[:, n_relevant:(2 * n_relevant)] = U
        
    
    
    
    
    # add Gaussian noise
    if noise and noise > 0:
        y = y + rng.normal(0, noise, size=n_samples)
    
    return X, y

#%%
R = 10
# o1neuro, xgb, rf, tabnet
optimization_time_all = [[] for _ in range(R)]
r_square_all = [[] for _ in range(R)]

run_times_o1neuro = [[] for _ in range(R)]
r_sqaures_o1neuro = [[] for _ in range(R)] # each consists of b = 100

params_o1neuro = [[] for _ in range(R)]

for r_ in range(R):
    
    #%%
    
    
    if type_ == 0:
        # Linaer Regression
        X_train, y_train = additive_model(weights, 
                                          n_features = p, 
                                          n_samples=n, 
                                          n_relevant = n_relevant)
        
        
    elif type_ == 2:
        # Multiplicative Interaction
        X_train, y_train = additive_model(weights,
                                          n_samples=n, 
                                          n_features = p,
                                          n_relevant = n_relevant, 
                                          make_xor=True)

    
    if type_ == 0:
        # Linaer Regression
        X_test, y_test = additive_model(weights, 
                                        n_samples=n_test,
                                        n_features = p,
                                        n_relevant = n_relevant, 
                                        noise=0)
    elif type_ == 2:
        # Multiplicative Interaction
        X_test, y_test =  additive_model(weights, 
                                         n_samples=n_test, 
                                         n_features = p,
                                         n_relevant = n_relevant, 
                                         make_xor=True, 
                                         noise=0)
    
    #%%
    print(f' sample size: {n}')
    if type_ == 2:
        print(f'Dataset type : {n_relevant} ( * 2) xor components')
    else:
        print(f'Dataset type : {n_relevant} linear components')
    validation_size = 0.2
    max_evals_ = 30
    X_train_in, X_train_out, y_train_in, y_train_out \
        = train_test_split(X_train, y_train, 
                           test_size = validation_size)
    
    
    #%%
    start_time = time.time()
    o1neuro_param, o1neuro = \
        o1Neuro_train(X_train_in,
                       X_train_out,
                       y_train_in,
                       y_train_out,
                       max_evals = max_evals_)
    end_time = time.time()
    optimization_time_all[r_].append(end_time - start_time)
    params_o1neuro[r_].append(o1neuro_param)
    
    
    print(f'used parameters : {o1neuro.get_params()}')
    for b_ in range(B):
        start_time = time.time()
        o1neuro.train(X_train,
                      y_train,
                      b = 1)
        end_time = time.time()
        run_times_o1neuro[r_].append(end_time - start_time)
        
        # Record 100 r squares of o1neuro
        y_pred_dspione = o1neuro.predict(X_test)
        r_square_o1neuro = 1 - mean_squared_error(y_pred_dspione, y_test) / np.var(y_test)
        r_sqaures_o1neuro[r_].append(r_square_o1neuro)
    
        print(f"R square o1D-Neuro: {r_square_o1neuro}, Round : {b_}")
        if b_ == min(4, B-1):
            r_square_all[r_].append(r_square_o1neuro)
    ###
    ###
    
    
    y_pred_dspione = o1neuro.predict(X_test)
    r_square_o1neuro = 1 - mean_squared_error(y_pred_dspione, y_test) / np.var(y_test)
        
        
    print(f"R square o1D-Neuro: {r_square_o1neuro}, Round : {r_}")
    
    
    
    #%%
    r_square_tabnet = 0
    runtime_tabnet = 0
    if run_tabnet:
        print(f'Run TabNet!')
        start_time = time.time()
        
        X_train_in_ = X_train_in.astype(np.float32)
        X_train_out_ = X_train_out.astype(np.float32)
        y_train_in_ = y_train_in.astype(np.float32)
        y_train_out_ = y_train_out.astype(np.float32)
        
        X_train_ = X_train.astype(np.float32)
        X_test_ = X_test.astype(np.float32)
        y_train_ = y_train.astype(np.float32)
        y_test_ = y_test.astype(np.float32)
        # max_evals_ = 10
        tabnet_param, tabnet = tabnet_train(X_train_in_,
                                            X_train_out_, 
                                            y_train_in_, 
                                            y_train_out_,
                                            max_evals = max_evals_)
        y_train_ = y_train_.reshape(-1, 1)
    
        # print(tabnet.__dict__)
        
    
    
    
    
        tabnet.fit(X_train_, 
                   y_train_,
                   patience=20,
                   max_epochs=200,
                   batch_size=tabnet_param["batch_size"],
                   virtual_batch_size= tabnet_param["virtual_batch_size"],
                   num_workers=0
                   )
        end_time = time.time()
    
        runtime_tabnet = end_time - start_time
                
        
        r_square_tabnet = 1 - mean_squared_error(tabnet.predict(X_test_), y_test) / np.var(y_test)
    
        print(f"R square TabNet: {r_square_tabnet}, Use runtime: {runtime_tabnet}")
    optimization_time_all[r_].append(runtime_tabnet)
    r_square_all[r_].append(r_square_tabnet)
    #%%
    
    r_square_cat = 0
    runtime_cat = 0
    if run_cat:
        start_time = time.time()
        cat_param, cat = cat_train(X_train_in,
                                   X_train_out, 
                                   y_train_in, 
                                   y_train_out,
                                   max_evals = max_evals_)
        end_time = time.time()
        
        runtime_cat = end_time - start_time
        cat.fit(X_train, y_train)
        
        
        # y_pred_dspione = xgbc.predict(X_test)
        r_square_cat = 1 - mean_squared_error(cat.predict(X_test), y_test) / np.var(y_test)
        
        print(f"R square CatBoost: {r_square_cat}")
    optimization_time_all[r_].append(runtime_cat)
    r_square_all[r_].append(r_square_cat)
    
    #%%
    r_square_xgb = 0
    runtime_xgb = 0
    if run_xgb:
        start_time = time.time()
        xgb_param, xgbc = xgb_train(X_train_in,
                                   X_train_out, 
                                   y_train_in, 
                                   y_train_out,
                                   max_evals = max_evals_)
        end_time = time.time()
        runtime_xgb = end_time - start_time
        
        xgbc.fit(X_train, y_train)
        
        
        # y_pred_dspione = xgbc.predict(X_test)
        r_square_xgb = 1 - mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test)
        
        print(f"R square XGB: {r_square_xgb}")
    optimization_time_all[r_].append(runtime_xgb)
    r_square_all[r_].append(r_square_xgb)
#%%
    r_square_rf = 0
    runtime_rf = 0
    if run_rf:
        start_time = time.time()
        rf_param, rf = rf_train(X_train_in,
                                X_train_out, 
                                y_train_in, 
                                y_train_out,
                                max_evals = max_evals_)
        rf.fit(X_train, y_train)
        end_time = time.time()
        runtime_rf = end_time - start_time
        
        # y_pred_cat = cat.predict(X_test)
        r_square_rf = 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test)
        
        print(f"R square RF: {r_square_rf}")
    
    optimization_time_all[r_].append(runtime_rf)
    r_square_all[r_].append(r_square_rf)
#%%
    if save_:
        print('saving files')
        path_temp = os.getcwd()
        result = path_temp.split("/")
        path = ''
        checker = True
        for elem in result:
            if elem != 'compact_o1neuro' and checker:
                path = path + elem + '/'
            else:
                checker = False
        path = path + 'compact_o1neuro/data' + '/'



        file = path + 'simulated_data/optimization_time_all' + str(n) + str(type_)
        with open(file, 'wb') as f:
            pickle.dump(optimization_time_all, f)
        
        file = path + 'simulated_data/params_o1neuro' + str(n) + str(type_)
        with open(file, 'wb') as f:
            pickle.dump(params_o1neuro, f)
        
        # o1Neuro, TabNet, CatBoost, XGBoost, RF
        file = path + 'simulated_data/r_square_all' + str(n) + str(type_)
        with open(file, 'wb') as f:
            pickle.dump(r_square_all, f)
        
        file = path + 'simulated_data/run_times_o1neuro' + str(n) + str(type_)
        with open(file, 'wb') as f:
            pickle.dump(run_times_o1neuro, f)
        
        file = path + 'simulated_data/r_sqaures_o1neuro' + str(n) + str(type_)
        with open(file, 'wb') as f:
            pickle.dump(r_sqaures_o1neuro, f)