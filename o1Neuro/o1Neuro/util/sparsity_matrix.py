#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:44:53 2025

@author: xbb
"""
import numpy as np


def sparsity_matrix_fast(X, s):
    """
    Vectorized version: Retain only `s` randomly chosen elements along axis=0
    for each K position in a (p0, K) matrix.
    """
    p0, K = X.shape
    if p0 > s:
        rand = np.random.rand(p0, K)
        top_s_idx = np.argpartition(rand, s, axis=0)[:s]  # shape: (s, K)



        # Initialize a mask of all True
        mask = np.ones_like(rand, dtype=bool)
    
        # Generate index grids for K to match shape (s, K)
        i2 = np.broadcast_to(np.arange(K)[None, :], (s, K))
        
        # Now set False values in one go
        mask[top_s_idx, i2] = False
        
        # Create sparse mask
        X[mask] = 0
    
    return X # Apply mask


