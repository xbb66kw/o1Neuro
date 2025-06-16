#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 21:09:45 2025

@author: xbb
"""



import os, pickle
import numpy as np


#%%


test_ = '300' # '300'
path_ = 'data/simulated_data/example' + str(test_)
path_time_ = 'data/simulated_data/runtime' + str(test_)

# Get the directory path for loading data_process_embryogrowth.rds
path_temp = os.getcwd()
result = path_temp.split("/")

path = ''
checker = True
for elem in result:
    if elem != 'bnn' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'bnn' + '/'
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.

#####
# Manually control for outputing summary results
# Codes include file reading commends

file = path + path_
with open(file, 'rb') as f:
    obj_rsquare_score = pickle.load(f)
obj_rsquare_score
# len(obj_rsquare_score[9])
# obj_rsquare_score[0][18]

file = path + path_time_
with open(file, 'rb') as f:
    obj_runtime = pickle.load(f)
obj_runtime

#%%
#####
# obj_rsquare_score is a list of length 10. Each records the 
# R^2 scores for all four methods (including the linear 
# regression) on each of the 19 datasets.
# See obj_rsquare_score[j], j = 0, ..., 18 for details.
if False:
    #%%
    #####
    # Across datasets comparison
    # average distance to the minimum (ADTM)
    R = 50  # number of repetition in the numerical experiments
    D_ = 2
    # Method; Dataset; Repetition
    result_table = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['r_square_o1neuro_bagging'],
                                       results[j]['xgb'],
                                       results[j]['rf']]

    score_all = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            M = np.max(result_table[:, j, ind])
            m = np.min(result_table[:, j, ind])
            for method in range(3):
                # Win rates
                score_all[method, j, ind] = \
                    (result_table[method, j, ind] - m) / (M - m)

  
    #%%
    # Report the detailed R^2 scores
    # Method; Dataset; Repetition
    result_table = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['r_square_o1neuro_bagging'],
                                       results[j]['xgb'],
                                       results[j]['rf']]
    
    ind_dataset = 0
    results = np.zeros((3, 3))
    results[:, 0] = np.max(result_table, axis=2)[:, ind_dataset]
    results[:, 1] = np.mean(result_table, axis=2)[:, ind_dataset]
    results[:, 2] = np.min(result_table, axis=2)[:, ind_dataset]    
    print(f'Model eqref(m1) (n=300) {results}')
    #%%
    ind_dataset = 1
    results = np.zeros((3, 3))
    results[:, 0] = np.max(result_table, axis=2)[:, ind_dataset]
    results[:, 1] = np.mean(result_table, axis=2)[:, ind_dataset]
    results[:, 2] = np.min(result_table, axis=2)[:, ind_dataset]
    print(f'Model eqref(m3) (n=300) {results}')
    
    # Print the results
    # ind_dataset = 0, 1, 2
    ind_dataset = 1
    # (boosted o1neuro, xgb, rf)
    print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
    print('std: ', np.std(result_table, axis=2)[:, ind_dataset])
    # np.mean(result_table[0])