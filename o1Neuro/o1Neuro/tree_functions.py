#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:17:43 2025

@author: xbb
"""

import numpy as np


from compact_o1neuro.util.param_search import search_start
from compact_o1neuro.util.parameter_tuning_space import \
    objective_rf_regression, space_rf, p_s_rf, \
    objective_o1neuro_regression, space_o1neuro, \
    objective_xgb_regression, space_xgb,\
    objective_tabnet_regression,space_tabnet

        
import xgboost as xgb


from sklearn.ensemble import RandomForestRegressor

import torch
from pytorch_tabnet.tab_model import TabNetRegressor


from compact_o1neuro.o1Neuro_jit import o1Neuro_jit


import warnings 
warnings.filterwarnings('ignore')


#%%
def rf_train(X_train_in, 
              X_train_out, 
              y_train_in,
              y_train_out,
              max_evals = 30): 
    
   

    best_param_rf = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_rf_regression,
        space_rf,
        max_evals = max_evals
        )


    
    best_param = best_param_rf
    rf = RandomForestRegressor(
        n_estimators=100,
        max_features=best_param['gamma'],
        max_depth=p_s_rf['max_depth'][best_param['max_depth']],                                    
        min_impurity_decrease=p_s_rf['min_impurity_decrease'][best_param['min_impurity_decrease']],
        min_samples_leaf=int(best_param['min_samples_leaf']),
        min_samples_split=int(best_param['min_samples_split']),
        criterion=p_s_rf['criterion'][best_param['criterion']])

    
    return best_param_rf, rf


#%%

def o1Neuro_train(X_train_in, 
                   X_train_out, 
                   y_train_in,
                   y_train_out,
                   max_evals = 30): 
    
   

    best_param_o1neuro = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_o1neuro_regression,
        space_o1neuro,
        max_evals = max_evals
        )
    

    
    best_param = best_param_o1neuro
    architecture = []

    
    n_layer = int(best_param['n_layer'])
    for i in range(int(best_param['n_layer'])):        
        n_neurons = int(best_param[str(n_layer) + '_layer'])
        if i < n_layer - 1:
            n_neurons = (n_neurons + 1)* 2**(n_layer - i - 1)
            architecture.append(n_neurons)
        else:
            architecture.append(n_neurons)
    

    o1neuro = o1Neuro_jit(architecture,
                          eta = best_param['eta'],
                          tuning_phase=False)

    o1neuro_param = {}
    
    o1neuro_param['architecture'] = architecture
    o1neuro_param['K'] = 20 # int(best_param['K'])
    o1neuro_param['eta'] = best_param['eta']
    return o1neuro_param, o1neuro


#%%


def xgb_train(X_train_in, 
              X_train_out, 
              y_train_in,
              y_train_out,
              max_evals = 30): 
    
   

    best_param_xgb = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_xgb_regression,
        space_xgb,
        max_evals = max_evals
        )
    
    

    
    best_param = best_param_xgb
    
    
    xgbc = xgb.XGBRegressor(
        n_estimators = 1000, 
        max_depth = int(best_param['max_depth']),
        gamma = np.exp(best_param['gamma']),
        reg_alpha = np.exp(best_param['reg_alpha']),
        reg_lambda = np.exp(best_param['reg_lambda']),
        min_child_weight = int(best_param['min_child_weight']),
        colsample_bytree = best_param['colsample_bytree'],
        colsample_bylevel = best_param['colsample_bylevel'],
        subsample = best_param['subsample'],
        learning_rate = np.exp(best_param['learning_rate']))
                   
           
    # xgbc.fit(X_train, y_train)
               
    
    
    return best_param_xgb, xgbc

#%%



def tabnet_train(X_train_in, 
                 X_train_out, 
                 y_train_in,
                 y_train_out,
                 max_evals = 30): 
    
   

    best_param_tabnet = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_tabnet_regression,
        space_tabnet,
        max_evals = max_evals
        )
    
    
    
    best_param = best_param_tabnet
    print(f'{best_param}')
    
    
    batch_size = [64, 128, 256, 512, 1024]
    best_param['batch_size'] = batch_size[best_param['batch_size']]
    
    virtual_batch_size = [64, 128]
    best_param['virtual_batch_size'] = virtual_batch_size[best_param['virtual_batch_size']]
    
    n_d = [8, 16, 24, 32]
    n_a = [8, 16, 24, 32]
    n_steps = [3, 4, 5]
    
    
    
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    tabnet = TabNetRegressor(
    n_d=n_d[best_param["n_d"]],
    n_a=n_a[best_param["n_a"]],
    n_steps=n_steps[best_param["n_steps"]],
    gamma=best_param["gamma"],
    lambda_sparse=best_param["lambda_sparse"],
    optimizer_fn=torch.optim.Adam,   # <- correct optimizer function
    optimizer_params=dict(lr=best_param["lr"]),
    verbose=0,
    device_name=device.type  # <--- IMPORTANT
    )
    
    

    
    
    return best_param, tabnet



