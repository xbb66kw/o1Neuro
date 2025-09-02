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


type_ = 2 # 0, 2
n = 20000 # 100, 500, 3000, 20000


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
path = path + 'compact_o1neuro/data/'
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.

#####
# Manually control for outputing summary results
# Codes include file reading commends




file = path + 'simulated_data/optimization_time_all' + str(n) + str(type_)
with open(file, 'rb') as f:
    optimization_time_all = pickle.load(f)
optimization_time_all

file = path + 'simulated_data/params_o1neuro' + str(n) + str(type_)
with open(file, 'rb') as f:
    params_o1neuro = pickle.load(f)
params_o1neuro


file = path + 'simulated_data/r_sqaures_o1neuro' + str(n) + str(type_)
with open(file, 'rb') as f:
    r_squares_o1neuro = pickle.load(f)
r_squares_o1neuro

file = path + 'simulated_data/r_square_all' + str(n) + str(type_)
with open(file, 'rb') as f:
    r_square_all = pickle.load(f)
r_square_all
# o1Neuro, TabNet, CatBoost, XGBoost, RF

file = path + 'simulated_data/run_times_o1neuro' + str(n) + str(type_)
with open(file, 'rb') as f:
    run_times_o1neuro = pickle.load(f)
run_times_o1neuro



# I have a list, named r_sqaures_o1neuro, of 10 list, each consisting 30 sequntial r2 scores from b = 1 to b = 30. I want to make a picture about it, icluding its mean across 10 lists agasint b, and the maximimum and minimum across 10 lists agasinst b/
#%%
#####
if False:
    #%%
    
    
    # Repeat R = 10 times.
    # o1neuro, xgboost, random forests, tabnet
    r2_mean = np.round(np.mean(r_square_all, axis=0), 3)
    r2_std = np.round(np.std(r_square_all, axis = 0), 3)
    
    # Combine mean and std into one string per model
    r2_report = [f"{m} ({s})" for m, s in zip(r2_mean, r2_std)]
    
    # Assuming r2_report = [o1Neuro, TabNet, CatBoost, XGBoost, RF]
    r2_report_filtered = [r2_report[i] for i in [0, 1, 3, 4]]
    print(f"R² results for Model Y{type_} for o1Neuro, TabNet, XGBoost, and Random Forests:\n" + " & ".join(r2_report_filtered))


#%% 
# Runtime
    # Assuming r2_report = [o1Neuro, TabNet, CatBoost, XGBoost, RF]

    # Model names without CatBoost
    model_names = ["o1Neuro", "TabNet", "XGBoost", "Random Forests"]

    run_time = np.mean(optimization_time_all, axis=0)
    # Exclude CatBoost (index 2)
    run_time_filled = [run_time[i] for i in [0, 1, 3, 4]]

    
    # Convert to NumPy array for division
    run_time_per_config = np.array(run_time_filled) / 30
    
    # Print all in one line
    print("Runtime per configuration (seconds): " + " | ".join(f"{model}: {rt:.2f}" for model, rt in zip(model_names, run_time_per_config)))

    
    print(f'Average run time for training a o1Neuro predictive model: {5*np.mean(run_times_o1neuro)}')
#%%
    
    # Convert list of lists to numpy array for easier computation
    # Shape: (10 runs, 30 b-values)
    r_squares_o1neuro = np.array(r_squares_o1neuro)

    # Compute statistics across runs
    r2_mean = np.mean(r_squares_o1neuro, axis=0)  # Mean R² across runs
    r2_min = np.min(r_squares_o1neuro, axis=0)    # Min R² across runs
    r2_max = np.max(r_squares_o1neuro, axis=0)    # Max R² across runs
    
    # Define x-axis: b = 1 to 30
    b_values = np.arange(1, r_squares_o1neuro.shape[1] + 1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(b_values, r2_mean, label='Mean R²', color='blue', linewidth=2)
    plt.fill_between(b_values, r2_min, r2_max, color='blue', alpha=0.1, label='Min–Max Range')
    

    # Set x-axis limit to cut at b = 30
    plt.xlim(1, 20)
    plt.ylim(0, 1)

    # Formatting
    # plt.title(f"Model {}", fontsize=14)
    plt.xlabel("Update Round (b)", fontsize=20)
    plt.ylabel("R² Score", fontsize=20)
    # plt.legend()
    
    # Put legend (caption) in bottom-right corner
    # plt.legend(loc='lower right', fontsize=20)
    plt.xticks(np.arange(1, 21, 1))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()