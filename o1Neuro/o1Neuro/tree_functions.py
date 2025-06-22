#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:17:43 2025

@author: xbb
"""

import numpy as np


from o1Neuro.util.param_search import search_start
from o1Neuro.util.parameter_tuning_space import \
    objective_rf_regression, space_rf, p_s_rf, \
    objective_o1neuro_regression, space_o1neuro, p_s_o1neuro, \
    objective_xgb_regression, space_xgb

        
import xgboost as xgb


from sklearn.ensemble import RandomForestRegressor
from o1Neuro.o1NeuroBoost import o1NeuroBoost


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

def o1neuroboost_train(X_train_in, 
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
    o1neuro = o1NeuroBoost(p_s_o1neuro['architecture'][best_param['architecture']][1:],
                           eta = best_param['eta'], 
                           M = int(best_param['M']),
                           K = 10, 
                           sparsity_level=2,
                           stochastic_ratio=best_param['stochastic_ratio'],
                           stabilizer= p_s_o1neuro['stabilizer'][best_param['stabilizer']])

 
    
    return best_param_o1neuro, o1neuro


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

