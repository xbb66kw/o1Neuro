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


# compact_o1neuro-0.1
from o1Neuro.tree_functions import rf_train, \
    xgb_train, o1Neuro_train, tabnet_train


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
run_tabnet = True
# save_ = True to save results
save_ = False



# test version
test_ = 'large' # 'large', 'beta', ''
make_xor = False # True, False

# Do not touch these parameters
test_size = 0.4
validation_size = 0.2



# number of optimization evaluations
max_evals_ = 1

# Repeat R times. Default R = 10
R = 1


obj_rsquare_score = [[] for r in range(R)]
obj_runtime = [[] for r in range(R)]
obj_rsquare_o1neuro = [[] for r in range(R)]

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


# Run beta veriosn with 5000 samples,
# and large version with 10000 samples.



# index : 
used_sample =  [2, 4, 13, 17]

start_ind_sample = 0
start_repeatition = 0  


#%%
if __name__ == '__main__':
    # for ind_sample in range(start_ind_sample, 19):
    for ind_sample in used_sample:
        file = path + 'openml/dataset/numpy' + str(ind_sample)
        with open(file, 'rb') as f:        
            dataset = pickle.load(f)   
        X_, y_, dataset = dataset
        dataset_name = dataset[45:].split('\n')[0]
        
        
        for q in range(start_repeatition, R):
            ind_rand = np.random.choice(np.arange(len(y_)), 
                    size = min(10000, len(y_)), replace = False)
            X = np.array(X_)[ind_rand, :]
            y = np.array(y_)[ind_rand]
            
            # Make features xor components:
            if make_xor:
                U = 2 * np.random.randint(0, 2, size=X.shape) - 1
                X_scaled = X / U
                # concatenate
                X = np.hstack([X_scaled, U])
                
                
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
            r_square_ = 0
            obj_rsquare_o1neuro_ = []
            if run_o1neuro:
                start_time = time.time()
                o1neuro_param, o1neuro = \
                    o1Neuro_train(X_train_in,
                                   X_train_out,
                                   y_train_in,
                                   y_train_out,
                                   max_evals = max_evals_)

                print(o1neuro.get_params())
                for b_ in range(5):
                    
                    o1neuro.train(X_train,
                                  y_train, 
                                  b = 1)
                
                    
                    ###
                    ###
                    end_time = time.time()
                    
                    y_pred_dspione = o1neuro.predict(X_test)
                    r_square_ = 1 - mean_squared_error(y_pred_dspione, y_test) / np.var(y_test)
                    
                    
                    r_square_o1neuro = 1 - mean_squared_error(y_pred_dspione, y_test) / np.var(y_test)
                    runtime_o1neuro = end_time - start_time                    
                    obj_rsquare_o1neuro_.append(r_square_o1neuro)
                    
                    print(f"R square o1D-Neuro: {r_square_}, Use runtime: {end_time - start_time}")
            
            
            r_square_tabnet = 0
            runtime_tabnet = 0
            if run_tabnet:
                start_time = time.time()
                
                X_train_in_ = X_train_in.astype(np.float32)
                X_train_out_ = X_train_out.astype(np.float32)
                y_train_in_ = y_train_in.astype(np.float32)
                y_train_out_ = y_train_out.astype(np.float32)
                
                X_train_ = X_train.astype(np.float32)
                X_test_ = X_test.astype(np.float32)
                y_train_ = y_train.astype(np.float32)
                y_test_ = y_test.astype(np.float32)
                
                tabnet_param, tabnet = tabnet_train(X_train_in_,
                                                    X_train_out_, 
                                                    y_train_in_, 
                                                    y_train_out_,
                                                    max_evals = max_evals_)
                y_train_ = y_train_.reshape(-1, 1)

                tabnet.fit(X_train_, 
                           y_train_,
                           patience=20,
                           max_epochs=200,
                           batch_size=tabnet_param["batch_size"],
                           virtual_batch_size= tabnet_param["virtual_batch_size"],
                           num_workers=0
                           )
                end_time = time.time()
    
                runtime_tabnet = end_time - start_time
                
                r_square_tabnet = 1 - mean_squared_error(tabnet.predict(X_test_), y_test) / np.var(y_test)
    
                print(f"R square TabNet: {r_square_tabnet}, Use runtime: {runtime_tabnet}")
                
                
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

            
            obj_rsquare_score[q].append({
                'r_square_o1neuro': r_square_o1neuro,
                'tabnet': r_square_tabnet,
                'xgb': r_square_xgb,
                'rf': r_square_rf,
                'dataset': dataset_name})
            

            obj_runtime[q].append({
                'runtime_o1neuro': runtime_o1neuro,
                'tabnet': runtime_tabnet,
                'xgb': runtime_xgb,
                'rf': runtime_rf,
                'dataset': dataset_name})
            
            obj_rsquare_o1neuro[q].append({
                'o1neuro_30_rounds': obj_rsquare_o1neuro_,
                'dataset': dataset_name})
            
            if save_:
                


                file = path + 'openml/results/r_square' + str(test_) + str(make_xor)
                print('Results for the best parameters for openml are \
                      saving at: ', '\n', file)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_score, f)
                file = path + 'openml/results/runtime' + str(test_) + str(make_xor)
                with open(file, 'wb') as f:
                    pickle.dump(obj_runtime, f)
                    
                file = path + 'openml/results/r_square_o1neuro' + str(test_) + str(make_xor)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_o1neuro, f)
                
                
                
        start_repeatition = 0
                
              
        
        