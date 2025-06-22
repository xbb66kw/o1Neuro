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
                1: "house_16H",
                2: "yprop_4_1",
                3: "abalone"}

#%% Load 



# beta version simulation
test_ = 'beta_square_large' # 'beta_square' 'beta_square_large'
path_ = 'data/openml/results/r_square' + str(test_)
path_time_ = 'data/openml/results/runtime' + str(test_)


# Get the directory path for loading data_process_embryogrowth.rds
path_temp = os.getcwd()
result = path_temp.split("/")

path = ''
checker = True
for elem in result:
    if elem != 'o1Neuro' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'o1Neuro' + '/'


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

#%%
#####
# obj_rsquare_score is a list of length 10. Each records the 
# R^2 scores for all four methods (including the linear 
# regression) on each of the 19 datasets.
# See obj_rsquare_score[j], j = 0, ..., 18 for details.
if False:
    #%%
    #####
    R = 10  # number of repetition in the numerical experiments
    D_ = 4
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

    # Print the overall results
    # method = 0 (o1neuro_bagging), 1 (xgb), 2 (rf)
    method = 2
    print(f' {np.round(np.max(np.mean(score_all[method], axis=0)), 3)} & {np.round(np.mean(score_all[method]), 3)}  & {np.round(np.min(np.mean(score_all[method], axis=0)), 3)}') 
    
    print('average winning rate:',
          np.mean(score_all[method]), '\n',
          'max wining rate: ',
          np.max(np.mean(score_all[method], axis=0)), '\n',
          'min wining rate: ',
          np.min(np.mean(score_all[method], axis=0)))
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

    # Print the results
    # ind_dataset = 0, ..., 18
    # ind_dataset = 3
    print('The R^2 scores for all three methods (o1neuro, xgb, rf)')

    # print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    # print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    # print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
    
    # up to 4 datasets
    for ind_dataset in range(4):
        a = np.max(result_table, axis=2)[:, ind_dataset]
        b = np.mean(result_table, axis=2)[:, ind_dataset]
        c = np.min(result_table, axis=2)[:, ind_dataset]
        d = np.std(result_table, axis=2)[:, ind_dataset]
        print(f'{dataset_name[ind_dataset]} \n {np.column_stack((a, b , c))}')
        print(f'std : {d}')
#%%
# Print runtime
# two datasets: house and superconditicity
    R = 10
    D_ = 4 # 2
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
