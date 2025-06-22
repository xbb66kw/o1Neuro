#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 21:09:45 2025

@author: xbb
"""



import os, pickle
import numpy as np
import matplotlib.pyplot as plt

#%%



test_ = '450' # '150' '450'
path_ = 'data/simulated_data/example' + str(test_)
path_time_ = 'data/simulated_data/runtime' + str(test_)
path_params_ = 'data/simulated_data/o1_params' + str(test_)

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
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.

#####
# Manually control for outputing summary results
# Codes include file reading commends

file = path + path_
with open(file, 'rb') as f:
    obj_rsquare_score = pickle.load(f)
obj_rsquare_score

file = path + path_params_
with open(file, 'rb') as f:
    obj_o1_params = pickle.load(f)
obj_o1_params

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
    D_ = 2 # number of model types 
    # Method; Dataset; Repetition
    result_table = np.zeros(4 * D_ * R).reshape(4, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['r_square_o1neuro'],
                                       results[j]['r_square_o1neuro_bagging'],
                                       results[j]['xgb'],
                                       results[j]['rf']]

    score_all = np.zeros(4 * D_ * R).reshape(4, D_, R)
    for j in range(D_):
        for ind in range(R):
            M = np.max(result_table[:, j, ind])
            m = np.min(result_table[:, j, ind])
            for method in range(3):
                # Win rates
                score_all[method, j, ind] = \
                    (result_table[method, j, ind] - m) / (M - m)

  
   
    #%%
    ind_dataset = 0 # 0, 1
    if ind_dataset == 1:
        model_ = 'xor model m3'
    else:
        model_ = 'linear model m1'
    results = np.zeros((4, 3))
    results[:, 0] = np.max(result_table, axis=2)[:, ind_dataset]
    results[:, 1] = np.mean(result_table, axis=2)[:, ind_dataset]
    results[:, 2] = np.min(result_table, axis=2)[:, ind_dataset]
    print(f'Model {model_} (n={test_}) ')
    
    # Print the results
    # ind_dataset = 0, 1, 2
    # ind_dataset = 1
    # (single boosted o1neuro, bagging boosted o1neuro, xgb, rf)
    print('single boosted o1neuro, bagging boosted o1neuro, xgb, rf')
    print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
    print('std: ', np.std(result_table, axis=2)[:, ind_dataset])
    # np.mean(result_table[0])
    
    
#%%

    
    
    ind_dataset = 0 # 0, 1
    architecture = [[4, 1], [8, 4, 1], [16, 8, 4, 1], [4, 4,1], [4,4,4,1], [4,4,4,4, 1]]
    arch1 = [0 for _ in range(len(architecture))]
    M1 = [0 for _ in range(50 - 3 + 1)]
    
    for r in range(R):
        ar_index = architecture.index(obj_o1_params[r][ind_dataset]['p'])
        arch1[ar_index] += 1
        M1[obj_o1_params[r][ind_dataset]['M'] - 3] += 1
    
    
    ind_dataset = 1 # 0, 1
    architecture = [[4, 1], [8, 4, 1], [16, 8, 4, 1], [4, 4,1], [4,4,4,1], [4,4,4,4, 1]]
    arch2 = [0 for _ in range(len(architecture))]
    M2 = [0 for _ in range(50 - 3 + 1)]
    
    for r in range(R):
        ar_index = architecture.index(obj_o1_params[r][ind_dataset]['p'])
        arch2[ar_index] += 1
        M2[obj_o1_params[r][ind_dataset]['M'] - 3] += 1
    
    


#%%
    
    
    # Example data
    np.random.seed(0)
    M1 = np.random.randint(3, 51, size=100)         # Top left
    arch1 = np.random.randint(1, 7, size=100)       # Top right
    M2 = np.random.randint(3, 51, size=100)         # Bottom left
    arch2 = np.random.randint(1, 7, size=100)       # Bottom right
        
    # Set up 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define integer-centered bin edges with padding for full bars
    bins_M = np.arange(2.5, 51.6, 1)     # Includes 3 to 50 fully
    bins_arch = np.arange(0.5, 6.6, 1)   # Includes 1 to 6 fully
    label_fontsize = 20
    label_fontsize_text = 10
    xtick_fontsize = 15
    ytick_fontsize = 15
    
    # Top Left: M1 (light blue)
    axs[0, 0].hist(M1, bins=bins_M, edgecolor='black', color='#8ea2E2')
    # axs[0, 0].set_xlabel('M')
    # axs[0, 0].set_ylabel('Model (10)', fontsize=label_fontsize)
    axs[0, 0].set_xticks(np.arange(3, 51, 5))
    axs[0, 0].tick_params(axis='x', labelsize=xtick_fontsize)
    axs[0, 0].tick_params(axis='y', labelsize=ytick_fontsize)

    
    # Top Right: arch1 (light red)
    axs[0, 1].hist(arch1, bins=bins_arch, edgecolor='black', color='#d0a080')
    # axs[0, 1].set_xlabel('Architecture')
    # axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_xticks(np.arange(1, 7, 1))
    axs[0, 1].tick_params(axis='x', labelsize=xtick_fontsize)
    axs[0, 1].tick_params(axis='y', labelsize=ytick_fontsize)
    axs[0, 1].set_xticks([1, 2, 3, 4, 5, 6])
    axs[0, 1].set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
    
    # Bottom Left: M2 (light blue)
    axs[1, 0].hist(M2, bins=bins_M, edgecolor='black', color='#8ea2E2')
    axs[1, 0].set_xlabel('M', fontsize=label_fontsize)
    # axs[1, 0].set_ylabel('Model (11)', fontsize=label_fontsize)
    axs[1, 0].set_xticks(np.arange(3, 51, 5))
    axs[1, 0].tick_params(axis='x', labelsize=xtick_fontsize)
    axs[1, 0].tick_params(axis='y', labelsize=ytick_fontsize)

    
    # Bottom Right: arch2 (light red)
    axs[1, 1].hist(arch2, bins=bins_arch, edgecolor='black', color='#d0a080')
    axs[1, 1].set_xlabel('Architecture', fontsize=label_fontsize)
    # axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xticks(np.arange(1, 7, 1))
    axs[1, 1].tick_params(axis='x', labelsize=xtick_fontsize)
    axs[1, 1].tick_params(axis='y', labelsize=ytick_fontsize)
    axs[1, 1].set_yticks([0, 5, 10, 15, 20])
    axs[1, 1].set_yticklabels(['0', '5', '10', '15', '20'])


    axs[1, 1].set_xticks([1, 2, 3, 4, 5, 6])
    axs[1, 1].set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
    
    plt.tight_layout()
    plt.show()