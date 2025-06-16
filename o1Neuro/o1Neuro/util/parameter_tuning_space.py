#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:45:07 2023

@author: xbb
"""
from typing import Any, Dict
import numpy as np




# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, hp
from sklearn.metrics import mean_squared_error
from hyperopt.pyll.base import scope
# Useful when debugging
import hyperopt.pyll.stochastic
# print(hyperopt.pyll.stochastic.sample(space))


import xgboost as xgb
xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0

from sklearn.ensemble import RandomForestRegressor
from o1Neuro.o1NeuroBoost import o1NeuroBoost






#%%
p_s_rf = {'max_depth': [None, 5, 10, 20, 50],
          'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
          'criterion': ['squared_error', 'absolute_error']}

space_rf = {
    'gamma': hp.uniform('gamma', 0, 1), 
    'max_depth': hp.choice('max_depth', p_s_rf['max_depth']),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', p_s_rf['min_impurity_decrease']),
    'criterion': hp.choice('criterion', p_s_rf['criterion'])}


def objective_rf_regression(space):

    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.


    clf = RandomForestRegressor(
        n_estimators=100,
        max_depth = space['max_depth'],
        max_features = space['gamma'],
        min_samples_leaf = int(space['min_samples_leaf']),
        min_samples_split = int(space['min_samples_split']),
        criterion = space['criterion'],
        min_impurity_decrease = space['min_impurity_decrease']
        )
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}

#%%
# p_s_o1neuro = {'architecture': [[0, 8, 4, 1], [0, 16, 8, 4, 1]]} 
p_s_o1neuro = {'architecture': [[0, 8, 1], [0, 8, 4, 1],
                                [0, 4, 1], [0, 4, 4, 1]]} 
# p_s_o1neuro = {'architecture': [[0, 8, 4, 1]]} 

space_o1neuro = {
    'eta': hp.uniform('eta', 0.05, 0.9), 
    'M': hp.quniform('M', 3, 50, 1),
    'b': hp.quniform('b', 99, 100, 1),
    'stochastic_ratio': hp.uniform('stochastic_ratio', 0.8, 1),
    'test_size' : hp.uniform('test_size', 0.1, 0.5), 
    'architecture': hp.choice('architecture', p_s_o1neuro['architecture'])}

def objective_o1neuro_regression(space):
    X_train_in, X_train_out, y_train_in, y_train_out = space['data']

    # it is a bug: an input list hyperparameter becomes a tuple
    snb = o1NeuroBoost(list(space['architecture'][1:]), 
                       eta = space['eta'], 
                       M = int(space['M']),
                       K = 10, 
                       sparsity_level=2,
                       stochastic_ratio = space['stochastic_ratio'])
    # print(f'space["b"] : {space["b"]}, space["M"] : {space["M"]},'
    #       f' eta : {space["eta"]}, space["architecture"] : {space["architecture"]}'
    #       f' test_size : {space["test_size"]}')
    
    snb.train_stable(X_train_in, 
                     y_train_in, 
                     b = int(space['b']), 
                     test_size = space["test_size"])


    prediction = snb.predict(X_train_out, bagging = False)

    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}
    
    
    #%%
# Tuning parameter space and objective functions for XGBoost

space_xgb = {
    'max_depth': scope.int(hp.quniform("max_depth", 2, 15, 1)),
    'gamma': hp.uniform('gamma', np.log(1e-8), np.log(7)),
    'reg_alpha': hp.uniform('reg_alpha', np.log(1e-8), np.log(1e2)),
    'reg_lambda': hp.uniform('reg_lambda', np.log(0.8), np.log(4)),
    'learning_rate': hp.uniform('learning_rate', np.log(1e-5), np.log(0.7)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
    'n_estimators': 1000}

# print(hyperopt.pyll.stochastic.sample(space_xgb))
# test_set = hyperopt.pyll.stochastic.sample(space_xgb)
# print(np.exp(test_set['gamma']), np.exp(test_set['learning_rate']), np.exp(test_set['reg_alpha']), np.exp(test_set['reg_lambda']), test_set['min_child_weight'])


def objective_xgb_regression(space):
    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
        
    
    model=xgb.XGBRegressor(
        n_estimators =space['n_estimators'], 
        max_depth = int(space['max_depth']), 
        gamma = np.exp(space['gamma']),
        reg_alpha = np.exp(space['reg_alpha']),
        reg_lambda = np.exp(space['reg_lambda']),
        learning_rate = np.exp(space['learning_rate']),
        min_child_weight=space['min_child_weight'],
        colsample_bytree=space['colsample_bytree'],
        colsample_bylevel = space['colsample_bylevel'],
        subsample = space['subsample'])
   

    
    # Define evaluation datasets.
    evaluation = [( X_train_in, y_train_in), 
                  ( X_train_out, y_train_out)]
    
    # Fit the model. Define evaluation sets, early_stopping_rounds,
    # and eval_metric.
    model.fit(X_train_in, y_train_in,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=20,verbose=False)

    # Obtain prediction and rmse score.
    pred = model.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, pred)
    
    # Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}

