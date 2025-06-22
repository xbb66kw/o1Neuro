#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:14:06 2023

@author: xbb
"""


import time
import warnings, os, pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import openml



from o1Neuro.tree_functions import rf_train, xgb_train, o1neuroboost_train

from o1Neuro.o1NeuroBoost import o1NeuroBoost

# Install this package first before usage.
# cd [...]/o1Neuro
# pip install -e .

###
### Restart the editor after the installation if needed.
###

#%%
#####
# Initialization
run_rf = True
run_xgb = True
run_o1neuro = True
# save_ = True to save results
save_ = True
continued_ = False


# test version
test_ = 'beta_square_large' # 'beta_square' # 'beta_square_large'


# Do not touch these parameters
test_size = 0.5
validation_size = 0.2



# number of optimization evaluations
max_evals_ = 100

# Repeat R times. Default R = 10
R = 10


obj_rsquare_score = [[] for r in range(R)]
obj_runtime = [[] for r in range(R)]

path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'o1Neuro' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'o1Neuro/data' + '/'

# elevator, house_16H, yprop_4_1, ablone
# index : 
used_sample = [2, 6, 16, 17]

start_ind_sample = 0
start_repeatition = 0  
if continued_:
    file_ = path + 'data/openml/results/r_square' + str(test_)
    print(f' Files saved at {file_}')
    with open(file_, 'rb') as f:
        obj_rsquare_score = pickle.load(f)
    
    
    
    content_length = []
    for content in obj_rsquare_score:
        content_length.append(len(content))
    if all(content_length - np.mean(content_length) == 0):
        start_ind_sample = content_length[0]
    elif any(content_length - np.mean(content_length) < 0):
        start_ind_sample = content_length[0] - 1
        start_repeatition = np.where(content_length - np.mean(content_length) < 0)[0][0]


#%%
if __name__ == '__main__':
    # start_ind_sample = 0
    # run time is reported for house and superconduct datasets
    # for ind_sample in [5, 15]:
    # for ind_sample in range(start_ind_sample, 19):
    for ind_sample in used_sample:
        file = path + 'openml/dataset/numpy' + str(ind_sample)
        with open(file, 'rb') as f:        
            dataset = pickle.load(f)   
        X_, y_, dataset = dataset
        dataset_name = dataset[45:].split('\n')[0]
        
        
        for q in range(start_repeatition, R):
            ind_rand = np.random.choice(np.arange(len(y_)), 
                    size = min(5000, len(y_)), replace = False)
            X = np.array(X_)[ind_rand, :]
            y = np.array(y_)[ind_rand]
            
            
                
            X_train, X_test, y_train, y_test\
                = train_test_split(X, y, test_size=test_size)
            
            
            print(X_train.shape)
            print(dataset)
            
            X_train_in, X_train_out, y_train_in, y_train_out \
                = train_test_split(X_train, y_train, 
                                   test_size = validation_size)
        
            # run o1Neuro
            r_square_o1neuro = 0
            runtime_o1neuro = 0
            if run_o1neuro:
                start_time = time.time()
                o1neuro_param, o1neuro = \
                    o1neuroboost_train(X_train_in,
                                       X_train_out,
                                       y_train_in,
                                       y_train_out,
                                       max_evals = max_evals_)

                o1neuro.train_reboost(X_train,
                                      y_train, 
                                      b = 120)
                                      # b = int`(o1neuro_param['b']))
                
                o1neuro.train_bagging(b = 10, n_estimators = 20)
                
                end_time = time.time()
                

                y_pred_dspione_bagging = o1neuro.predict(X_test)
                r_square_o1neuro_bagging = 1 - mean_squared_error(y_pred_dspione_bagging, y_test) / np.var(y_test)
                runtime_o1neuro = end_time - start_time
                
                
                
                print(f"R square bagging o1Neuro: {r_square_o1neuro_bagging}, "
                      f"R square o1Neuro: {r_square_o1neuro}, Use runtime: {end_time - start_time}")
                
            
            # run xgb
            r_square_xgb = 0
            runtime_xgb = 0
            if run_xgb:
                start_time = time.time()
                xgb_param, xgbc = xgb_train(X_train_in, 
                                           X_train_out, 
                                           y_train_in, 
                                           y_train_out,
                                           max_evals = max_evals_)
                xgbc.fit(X_train, y_train)
                end_time = time.time()
                
                y_pred_dspione = xgbc.predict(X_test)
                r_square_xgb = 1 - mean_squared_error(xgbc.predict(X_test), y_test) / np.var(y_test)
                runtime_xgb = end_time - start_time
                print(f"R square XGB: {r_square_xgb}, Use runtime: {end_time - start_time}")
                
            # run Random Forests
            r_square_rf = 0
            runtime_rf = 0
            if run_rf:
                start_time = time.time()
                rf_param, rf = rf_train(X_train_in, 
                                        X_train_out, 
                                        y_train_in, 
                                        y_train_out,
                                        max_evals = max_evals_)
                # rf.get_params()
                rf.fit(X_train, y_train)
                end_time = time.time()
                
                y_pred_dspione = rf.predict(X_test)
                r_square_rf = 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test)
                runtime_rf = end_time - start_time
                
                
                
                print(f"R square RF: {r_square_rf}, Use runtime: {end_time - start_time}")
                
             #####          

            
                    
            if save_:
                obj_rsquare_score[q].append({
                    'r_square_o1neuro_bagging': r_square_o1neuro_bagging,
                    'xgb': r_square_xgb,
                    'rf': r_square_rf,
                    'dataset': dataset_name})
                

                obj_runtime[q].append({
                    'runtime_o1neuro': runtime_o1neuro,
                    'xgb': runtime_xgb,
                    'rf': runtime_rf,
                    'dataset': dataset_name})


                file = path + 'openml/results/r_square' + str(test_)
                print('Results for the best parameters for openml are \
                      saving at: ', '\n', file)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_score, f)
                file = path + 'openml/results/runtime' + str(test_)
                with open(file, 'wb') as f:
                    pickle.dump(obj_runtime, f)
        start_repeatition = 0
                
              
        
        