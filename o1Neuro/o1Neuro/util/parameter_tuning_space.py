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

import torch
from pytorch_tabnet.tab_model import TabNetRegressor


from compact_o1neuro.o1Neuro_jit import o1Neuro_jit







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



space_o1neuro = {
    'eta': hp.uniform('eta', 0.0, 0.6),
    'b': hp.quniform('b', 5, 6, 1), # 30
    'n_layer':  hp.quniform('n_layer', 1, 2, 1),
    '1_layer':  hp.quniform('1_layer', 800, 1000, 1),
    '2_layer':  hp.quniform('2_layer', 450, 650, 1),
    # '3_layer':  hp.quniform('3_layer', 400, 500, 1),
    # '4_layer':  hp.quniform('4_layer', 400, 500, 1),
}

def objective_o1neuro_regression(space):
    X_train_in, X_train_out, y_train_in, y_train_out = space['data']

    architecture = []
    n_neurons = float('Inf')

    n_layer = int(space['n_layer'])
    for i in range(n_layer):
        n_neurons = int(space[str(n_layer) + '_layer'])
        if i < n_layer - 1:
            n_neurons = (n_neurons + 1)* 2**(n_layer - i - 1)
            architecture.append(n_neurons)
        else:
            architecture.append(n_neurons)
        
    # The use of dropout and stochastic_ratio spped up the
    # hyperparameter optimization procedure
    model = o1Neuro_jit(architecture,
                        eta = space['eta'],
                        tuning_phase=True)


    
    model.train(X_train_in, 
                y_train_in, 
                b = int(space['b']))
    prediction = model.predict(X_train_out)


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


#%%

space_tabnet = {
    # Lower dimensions for faster training, still expressive enough
    "n_d": hp.choice("n_d", [8, 16, 24]),
    "n_a": hp.choice("n_a", [8, 16, 24]),
    
    # Fewer steps → less computation, still captures interactions
    "n_steps": hp.choice("n_steps", [3, 4, 5]),
    
    # Keep gamma tighter, avoid very high values that slow convergence
    "gamma": hp.uniform("gamma", 1.0, 1.8),
    
    # Regularization range tightened → avoids overfitting but less tuning overhead
    "lambda_sparse": hp.loguniform("lambda_sparse", np.log(1e-5), np.log(1e-2)),
    
    # Learning rate range narrowed for stability and faster convergence
    "lr": hp.loguniform("lr", np.log(1e-3), np.log(0.02)),
    
    # Use larger batch sizes for GPU/MPS acceleration
    "batch_size": hp.choice("batch_size", [64, 128, 256, 512]),
    
    # Virtual batch size must be much smaller than batch size for efficiency
    "virtual_batch_size": hp.choice("virtual_batch_size", [64, 128]),
}

def objective_tabnet_regression(space):
    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
        
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    
    model = TabNetRegressor(
    n_d=space["n_d"],
    n_a=space["n_a"],
    n_steps=space["n_steps"],
    gamma=space["gamma"],
    lambda_sparse=space["lambda_sparse"],
    optimizer_fn=torch.optim.Adam,   # <- correct optimizer function
    optimizer_params=dict(lr=space["lr"]),
    verbose=0,
    device_name=device.type  # <--- IMPORTANT
    )
    
    # Reshape y into (n_samples, 1)
    y_train_in = y_train_in.reshape(-1, 1)
    y_train_out = y_train_out.reshape(-1, 1)
    
    model.fit(
    X_train_in, 
    y_train_in,
    # eval_set=[(X_train_out, y_train_out)],
    # eval_metric=["rmse"],
    patience=20,
    max_epochs=200,
    batch_size=space["batch_size"],
    virtual_batch_size=space["virtual_batch_size"],
    num_workers=0,
    )
    
    # Fit the model. Define evaluation sets, early_stopping_rounds,
    # and eval_metric.
    # model.fit(X_train_in, y_train_in, eval_set=(X_train_out, y_train_out), use_best_model=True)
    model.fit(X_train_in, y_train_in)
   

    # Obtain prediction and rmse score.
    pred = model.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, pred)
    
    # Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}



