#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:45:57 2023

@author: xbb
"""

import os, pickle
import numpy as np

#%%



dataset_name = {0: "elevators",
                1: "Alieron",
                2: "medical\_charge",
                3: "abalone"}

#%% Load 



# beta version simulation
test_ = 'large' # 'large', 'beta', ''
make_xor = False # True, False
path_ = 'data/openml/results/r_square' + str(test_) + str(make_xor)
path_time_ = 'data/openml/results/runtime' + str(test_) + str(make_xor)
path_o1neuro = 'data/openml/results/r_square_o1neuro' + str(test_) + str(make_xor)



# Get the directory path for loading data_process_embryogrowth.rds
path_temp = os.getcwd()
result = path_temp.split("/")

path = ''
checker = True
for elem in result:
    if elem != 'compact_o1neuro' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'compact_o1neuro/'
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.


#####
# Manually control for outputing summary results
# Codes include file reading commends

file = path + path_
with open(file, 'rb') as f:
    obj_rsquare_score = pickle.load(f)
obj_rsquare_score


file = path + path_time_
with open(file, 'rb') as f:
    obj_runtime = pickle.load(f)
obj_runtime

file = path + path_o1neuro
with open(file, 'rb') as f:
    obj_rsquare_o1neuro = pickle.load(f)
obj_rsquare_o1neuro


#%%
#####
# obj_rsquare_score is a list of length 10. Each records the 
# R^2 scores for all four methods (including the linear 
# regression) on each of the 19 datasets.
# See obj_rsquare_score[j], j = 0, ..., 18 for details.
if False:   
    #%%
    R = 10  # number of repetition in the numerical experiments
    D_ = 4
    # Report the detailed R^2 scores
    # Method; Dataset; Repetition
    result_table = np.zeros(4 * D_ * R).reshape(4, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['r_square_o1neuro'],
                                       results[j]['tabnet'],
                                       results[j]['xgb'],
                                       results[j]['rf']]

    # Print the results
    # ind_dataset = 0, ..., 18
    # ind_dataset = 3
    print('The R^2 scores for all three methods at row (o1neuro, xgb, rf)')

    # print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    # print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    # print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
    
    # up to 4 datasets
    names = ["o1Neuro", "TabNet",  "XGBoost", "RF"]
    for ind_dataset in range(D_):
        a = np.max(result_table, axis=2)[:, ind_dataset]
        b = np.mean(result_table, axis=2)[:, ind_dataset]
        c = np.min(result_table, axis=2)[:, ind_dataset]
        d = np.std(result_table, axis=2)[:, ind_dataset]
        # print(f'{dataset_name[ind_dataset]} \n {np.column_stack((a, b , c))}')
        # print(f'std : {d}')
        
        A = np.column_stack((a, b , c))


        # Generate LaTeX rows
        for i, name in enumerate(names):
            row = f"{name:13} & {np.max(A[i]):.3f} & {np.mean(A[i]):.3f} ({d[i]:.3f}) & {np.min(A[i]):.3f} \\\\"
            print(row)
        print('\n')

#%%
# Print runtime
# two datasets: 
    R = 10
    D_ = 4 # 3
    runtime_set = [0, 1, 2, 3]
    result_table = np.zeros(3 * D_ * R).reshape(3, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_runtime[ind]
            result_table[:, j, ind] = [results[j]['runtime_o1neuro'],
                                       results[j]['xgb'],
                                       results[j]['rf']]



    for ind_dataset in range(D_):
        a = np.round(np.max(result_table, axis=2)[:, ind_dataset])
        b = np.round(np.mean(result_table, axis=2)[:, ind_dataset])
        c = np.round(np.min(result_table, axis=2)[:, ind_dataset])
        print(f'{dataset_name[runtime_set[ind_dataset]]} \n {np.column_stack((a, b, c))}')
