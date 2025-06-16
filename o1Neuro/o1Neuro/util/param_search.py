#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:26:51 2023

@author: xbb
"""

# from sklearn.model_selection import train_test_split
from hyperopt import Trials, fmin, tpe
    
def search_start(X_train_in, 
                  X_train_out, 
                  y_train_in, 
                  y_train_out,
                  objective_regression,
                  space,
                  n_trees_user: int | None = None,
                  max_evals: int = 100) -> dict:
    
    
    
    space['data'] = (X_train_in, X_train_out, y_train_in, y_train_out)
    
    trials = Trials()
    best_hyperparams = fmin(fn = objective_regression,
                            space = space,
                            trials = trials,
                            algo = tpe.suggest,
                            max_evals = max_evals)

    del space['data']

    return best_hyperparams
