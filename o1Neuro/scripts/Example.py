#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 12:40:54 2025

@author: xbb
"""



import numpy as np
import warnings, os, pickle
import time


# conda
# pip3 install torch torchvision torchaudio


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# from o1NeuroBoost import o1NeuroBoost
from o1Neuro.tree_functions import rf_train, xgb_train, o1neuroboost_train

from o1Neuro.o1NeuroBoost import o1NeuroBoost
# import cupy as cp
#%%

run_o1neuro = True
run_xgb = True
run_rf = True
save_ = True

# number of optimization evaluations
max_evals_ = 30

# Repeat R times. Default R = 10
R = 50


test_ = '' # ''
n = 450 # 150,  450
n_test = 10000
p = 100


obj_o1neuro_params = [[] for _ in range(R)]
obj_rsquare_score = [[] for _ in range(R)]
obj_runtime = [[] for _ in range(R)]




path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'o1Neuro' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'o1Neuro/data' + '/'
#%%

for type_ in [0, 2]:
    for r in range(R):
        #%%
        # Generate a sample of size n    
    
        X_train = np.random.uniform(-1, 1, size=(n, p))
        if type_ == 0:
            # Linaer Regression
            # signal-to-noise ratio approximately 1:1
            y_train = X_train[:, 0] + X_train[:, 1] + X_train[:, 2] + X_train[:, 3] + X_train[:, 4] + \
                X_train[:, 5] + X_train[:, 6] + X_train[:, 7] + X_train[:, 8] + X_train[:, 9] + 1.8 * np.random.randn(n)
        elif type_ == 2:
            # signal-to-noise ratio approximately 5:1:1
            # Multiplicative Interaction
            y_train = 2 * X_train[:, 0] * X_train[:, 1] + 0.5 * X_train[:, 2] + 0.3 * np.random.randn(n)
        

        
        X_test = np.random.uniform(-1, 1, size=(n_test, p))
        
        if type_ == 0:
            # Linaer Regression
            y_test = X_test[:, 0] + X_test[:, 1] + X_test[:, 2] + X_test[:, 3] + X_test[:, 4] + \
                X_test[:, 5] + X_test[:, 6] + X_test[:, 7] + X_test[:, 8] + X_test[:, 9]
        elif type_ == 2:
            # Multiplicative Interaction
            y_test = 2 * X_test[:, 0] * X_test[:, 1] + 0.5 * X_test[:, 2]
        
        
        #%%
        
        
        
        
        
        
        print(X_train.shape)
        print(type_)
        
        X_train_in, X_train_out, y_train_in, y_train_out \
            = train_test_split(X_train, y_train, 
                               test_size = 0.2)
        
        # run o1Neuro
        r_square_o1neuro = 0
        runtime_o1neuro = 0
        if run_o1neuro:
            start_time = time.time()
            o1neuro_param, o1neuro = \
                o1neuroboost_train(X_train_in,
                                   X_train_out,
                                   y_train_in,
                                   y_train_out,
                                   max_evals = max_evals_)
            o1neuro.train_reboost(X_train,
                                 y_train, 
                                 b = 120)
                                 # b = int(o1neuro_param['b']))
            print(f'selected params : {o1neuro_param}')
            o1neuro.train_bagging(b = 10, n_estimators = 20)
            
            ###
            
            ###
            end_time = time.time()
            
            
            o1_params = o1neuro.get_params()
        
            y_pred = o1neuro.predict(X_test)
            r_square_o1neuro_bagging = 1 - mean_squared_error(y_pred, y_test) / np.var(y_test)

            y_pred = o1neuro.predict(X_test, bagging = False)
            r_square_o1neuro = 1 - mean_squared_error(y_pred, y_test) / np.var(y_test)

            
            
            
            runtime_o1neuro = end_time - start_time
            
            
            
            print(f"R square bagging o1Neuro: {r_square_o1neuro_bagging}, "
                  f"R square o1Neuro: {r_square_o1neuro}, Use runtime: {end_time - start_time}")
            
        
        # run xgb
        r_square_xgb = 0
        runtime_xgb = 0
        if run_xgb:
            start_time = time.time()
            xgb_param, xgbc = xgb_train(X_train_in, 
                                       X_train_out, 
                                       y_train_in, 
                                       y_train_out,
                                       max_evals = max_evals_)
            xgbc.fit(X_train, y_train)
            end_time = time.time()
            
            y_pred_dspione = xgbc.predict(X_test)
            r_square_xgb = 1 - mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test)
            runtime_xgb = end_time - start_time
            print(f"R square XGB: {r_square_xgb}, Use runtime: {end_time - start_time}")
            
        # run Random Forests
        r_square_rf = 0
        runtime_rf = 0
        if run_rf:
            start_time = time.time()
            rf_param, rf = rf_train(X_train_in, 
                                    X_train_out, 
                                    y_train_in, 
                                    y_train_out,
                                    max_evals = max_evals_)
            # rf.get_params()
            rf.fit(X_train, y_train)
            end_time = time.time()
            
            y_pred_dspione = rf.predict(X_test)
            r_square_rf = 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test)
            runtime_rf = end_time - start_time
            
            
            
            print(f"R square RF: {r_square_rf}, Use runtime: {end_time - start_time}")
            
        
        if save_:
            obj_o1neuro_params[r].append(o1_params)
            obj_rsquare_score[r].append({
                'r_square_o1neuro': r_square_o1neuro,
                'r_square_o1neuro_bagging': r_square_o1neuro_bagging,                
                'xgb': r_square_xgb,
                'rf': r_square_rf,
                'dataset': type_})
            
        
            obj_runtime[r].append({
                'runtime_o1neuro': runtime_o1neuro,
                'xgb': runtime_xgb,
                'rf': runtime_rf,
                'dataset': type_})
        
        
            file = path + 'simulated_data/example' + str(n) + test_
            print('Results for the best parameters for openml are \
                  saving at: ', '\n', file)
            with open(file, 'wb') as f:
                pickle.dump(obj_rsquare_score, f)
            file = path + 'simulated_data/runtime' + str(n) + test_
            with open(file, 'wb') as f:
                pickle.dump(obj_runtime, f)
                
            file = path + 'simulated_data/o1_params' + str(n) + test_
            with open(file, 'wb') as f:
                pickle.dump(obj_o1neuro_params, f)
            