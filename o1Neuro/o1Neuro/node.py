from __future__ import division
import numpy as np
class Node(object):
    def __init__(self,
                 X: np.ndarray,
                 feature_index: int | list[int] | None,
                 threshold: float | None,
                 ind_set: np.ndarray,
                 depth: int,
                 node_index: int,
                 ave) -> None:
        # `feature_index` may be None, which means the Node has
        # not been split yet.
        self.feature_index = feature_index
        self.threshold = threshold
        self.ind_set = ind_set
        # self.tree_index = tree_index
        self.node_index = node_index
        self.ave = ave
        self.depth = depth
        
        # A parent node will be assigned immediately after
        # a Node is created.
        self.parent: Node | None = None
        
        # Initial importance measure is zero.
        self.importance: float = 0
        
        self.children: list[Node] = []
        self.stop = False
        
        # The 0th axis, 1st axis, ...
        # From outside of the list recursion to inside of the list.
        
        X_temp = X[self.ind_set, :]
        
        # argsort with axis = 0 sorting X_temp along the 0th axis
        self.X_arg_temp = np.argsort(X_temp, axis = 0)\
            .reshape([X_temp.shape[0] * X_temp.shape[1], ])
       
        ######
        # The recording of instances where features exhibit 
        # repeated values.        
        X_sort_temp = np.sort(X_temp, axis = 0)
        self.ind_repeats = X_sort_temp[1:] <= X_sort_temp[:-1]
        self.ind_repeats = self.ind_repeats.transpose()\
            .reshape([(X_temp.shape[0] - 1) * X_temp.shape[1], ])
        ######
        
        # Print self.seq1 to see its structure
        self.seq1 = np.repeat(a = np.arange(\
            X_temp.shape[0] -1) + 1, repeats = X_temp.shape[1])\
            .reshape(X_temp.shape[0] - 1, X_temp.shape[1])
        self.seq2 = X_temp.shape[0] - self.seq1

    def cal_loss(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 y_residual: np.ndarray,
                 y_residual_s: np.ndarray) -> np.ndarray:
        '''
        

        Parameters
        ----------
        X : np.ndarray

        y : np.ndarray

        y_residual : np.ndarray
            
        y_residual_s : np.ndarray
            Squared residuals.
            
        group_info : list[list[int]] | list
            A list of indices of features in groups. The default 
            is [].

        Returns
        -------
        loss_reduction : np.ndarray
            Calculate a vector of loss redutions.

        '''

        # Load and shorten variables
        ind_set = self.ind_set
        X_arg_temp = self.X_arg_temp
        seq1 = self.seq1
        seq2 = self.seq2
        X_temp = X[ind_set, :]
        L = X_temp.shape[0]

        y_residual = y_residual[ind_set]
        y_residual_s = y_residual_s[ind_set]
        

        y_temp = y_residual[X_arg_temp].reshape([L, X_temp.shape[1]])
        y_temp_s = y_residual_s[X_arg_temp].reshape([L, X_temp.shape[1]])
            

        y_cumsum = np.cumsum(y_temp, axis = 0)
        y_sum = y_cumsum[L - 1]
        y_cumsum = y_cumsum[:(L - 1)]

        y_s_cumsum = np.cumsum(y_temp_s, axis = 0)
        y_s_sum = y_s_cumsum[L - 1]
        y_s_cumsum = y_s_cumsum[:(L - 1)]

        y_cumsum_rev = y_sum - y_cumsum
        y_s_cumsum_rev = y_s_sum - y_s_cumsum

        y_total = y_s_sum

        # The formula is given in the paper
        # Speed up when predicting a univariate response
        A = y_total \
                - y_s_cumsum_rev + (y_cumsum_rev**2) / seq2 \
                - y_s_cumsum + (y_cumsum**2) / seq1
        loss_reduction = A.transpose()\
            .reshape([(L - 1) * X_temp.shape[1], ])
        loss_reduction[self.ind_repeats] = -float('Inf')
        

        return loss_reduction

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        

        Parameters
        ----------
        X : np.ndarray            

        Returns
        -------
        y : np.ndarray
            Predict the vector-valued mean of the corresponding
            response of each sample in X.

        '''
        num_samples = X.shape[0]
        y = np.empty(num_samples)

        for i in range(num_samples):
            node = self
            while node.threshold is not None:
                if X[i][node.feature_index] <= node.threshold:
                    # Xj values are small
                    node = node.children[0] 
                else:
                    # Xj values are large
                        node = node.children[1]
            y[i] = node.ave
        return y
    