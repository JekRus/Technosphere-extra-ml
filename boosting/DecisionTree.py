import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class Tree:
    def __init__(self):
        self.depth = 0
        self.feature_ind = 0
        self.threshold_ind = 0
        self.threshold = 0.0
        self.prediction = None
        self.left = None
        self.right = None


class Decision_tree(BaseEstimator, RegressorMixin):
    """My Decision tree"""
    
    def __init__(self, max_depth=None, min_leaf_size=1, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_impurity = min_impurity
        self.tree = Tree()
    
    def presort(self, X):
        return np.argsort(X, axis=0).T
    
    def mse(self, y):
        return 1.0/float(y.size) * np.sum((y - np.mean(y))**2)
    
    def stop_criterion(self, tree, y):
        if self.max_depth is not None and tree.depth == self.max_depth:
            return True
        if y.size <= self.min_leaf_size:
            return True
        if len(np.unique(y)) == 1:
            return True
        if self.mse(y) <= self.min_impurity:
            return True
        return False
    
    def best_split(self, X, y, sorted_ind):
        max_gain = 0.0
        best_feature_ind = 0
        best_threshold = 0
        best_threshold_ind = 0
        y_sum = np.sum(y)
        y_sum_sq = np.sum(y**2)
        y_size = y.size
        y_impurity = (y_sum_sq - y_sum**2/float(y_size))/float(y_size)
        
        for feature_ind in xrange(X.shape[1]):
            feature_sorted = X.T[feature_ind][sorted_ind[feature_ind]]
            y_sorted = y[sorted_ind[feature_ind]]
            left_size, right_size = 0, y_sorted.size
            left_sum = left_sum_sq = 0.0
            right_sum = y_sum
            right_sum_sq = y_sum_sq
            
            for t_ind in xrange(feature_sorted[:-1].shape[0]):
                left_sum += y_sorted[t_ind]
                right_sum -= y_sorted[t_ind]
                left_sum_sq += y_sorted[t_ind]**2
                right_sum_sq -= y_sorted[t_ind]**2
                left_size += 1
                right_size -= 1
                
                if feature_sorted[t_ind] == feature_sorted[t_ind + 1]:
                    continue
                if y_sorted[t_ind] == y_sorted[t_ind+1]:
                    continue
                
                left_impurity = left_sum_sq - 1.0/float(left_size) * left_sum**2
                right_impurity = right_sum_sq - 1.0/float(right_size) * right_sum**2
                information_gain = y_impurity - 1.0/float(y_size) * (left_impurity + right_impurity)
                
                if information_gain > max_gain:
                    max_gain = information_gain
                    best_feature_ind = feature_ind
                    best_threshold = (feature_sorted[t_ind] + feature_sorted[t_ind + 1])/2.0
                    best_threshold_ind = t_ind
                    
        return [max_gain, best_feature_ind, best_threshold, best_threshold_ind]
    
    def build(self, X, y, sorted_ind, depth=0):
        ##create node t
        tree = Tree()
        tree.depth = depth
        if self.stop_criterion(tree, y):
            ##assign a predictive model to tree
            tree.prediction = np.mean(y)
        else:
            ##Find the best binary split L = L_left + L_right
            gain, tree.feature_ind, tree.threshold, tree.threshold_ind = self.best_split(X, y, sorted_ind)
            
            left = np.where(X.T[tree.feature_ind] <= tree.threshold)[0]
            right = np.where(X.T[tree.feature_ind] > tree.threshold)[0]
            index_list = [None] * X.shape[0]
            for i in xrange(left.shape[0]):
                index_list[left[i]] = (i, "l")
            for i in xrange(right.shape[0]):
                index_list[right[i]] = (i, "r")
            
            sorted_ind_left = np.zeros((self.X.shape[1], left.shape[0])).astype(int)
            sorted_ind_right = np.zeros((self.X.shape[1], right.shape[0])).astype(int)
            
            for f_ind in xrange(X.shape[1]):
                sorted_1d = sorted_ind[f_ind]
                tmp_l = []
                tmp_r = []
                for idx in xrange(sorted_1d.shape[0]):
                    if index_list[sorted_1d[idx]][1] == 'l':
                        tmp_l.append(index_list[sorted_1d[idx]][0])
                    else:
                        tmp_r.append(index_list[sorted_1d[idx]][0])
                sorted_ind_left[f_ind] = np.array(tmp_l)
                sorted_ind_right[f_ind] = np.array(tmp_r)
                    
            tree.left = self.build(X[left], y[left], sorted_ind_left, tree.depth+1)
            tree.right = self.build(X[right], y[right], sorted_ind_right, tree.depth+1)
        return tree
    
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        sorted_ind = self.presort(X)
        self.tree = self.build(X, y, sorted_ind)
        return
    

    def evaluate_tree(self, X, tree):
        if tree.left == None:
            return tree.prediction * np.ones(X.shape[0])
        left_index = X.T[tree.feature_ind] <= tree.threshold
        right_index = np.invert(left_index)
        
        left_prediction = self.evaluate_tree(X[left_index], tree.left)
        right_prediction = self.evaluate_tree(X[right_index], tree.right)
        prediction = np.zeros(X.shape[0])
        prediction[left_index] = left_prediction
        prediction[right_index] = right_prediction
        return prediction
        
    
    def predict(self, X):
        prediction = self.evaluate_tree(X, self.tree)
        return prediction
