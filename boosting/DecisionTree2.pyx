import numpy as np
cimport numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import cython
from libc.math cimport abs


class Tree:
    def __init__(self):
        self.depth = 0
        self.feature_ind = 0
        self.threshold_ind = 0
        self.threshold = 0.0
        self.gain = 1.0
        self.samples = None
        self.mse = None
        self.prediction = None
        self.left = None
        self.right = None


class Decision_tree_:
    """My Decision tree"""
    
    def __init__(self, int max_depth=10000, int min_samples_split=2, double min_impurity=1e-50, double min_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.min_gain = min_gain
        self.tree = None
    
    def presort(self, np.ndarray[np.float64_t, ndim=2] X):
        return np.asfortranarray(np.argsort(X, axis=0).T)
    
    def mse(self, np.ndarray[np.float64_t, ndim=1] y):
        return 1.0/float(y.size) * np.sum((y - np.mean(y))**2)
    
    def stop_criterion(self, tree, np.ndarray[np.float64_t, ndim=1] y):
        if tree.depth == self.max_depth:
            return True
        if y.size < self.min_samples_split:
            return True
        if len(np.unique(y)) == 1:
            return True
        if tree.gain <= self.min_gain:
            return True
        if self.mse(y) <= self.min_impurity:
            return True
        return False
    
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def best_split(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y, 
                    np.ndarray[long, ndim=2] sorted_ind):
        ##basic initialization
        cdef int best_feature_ind, best_threshold_ind, y_size
        cdef double max_gain, best_threshold, y_sum, y_sum_sq, y_impurity
        
        max_gain = 0.0
        best_feature_ind = 0
        best_threshold = 0.0
        best_threshold_ind = 0
        y_sum = 0.0
        y_sum_sq = 0.0
        y_size = y.size
        #precalculate sum, sum of squares and impurity of the whole set
        cdef int i
        for i in xrange(y_size):
            y_sum += y[i]
            y_sum_sq += y[i]**2
        
        y_impurity = (y_sum_sq - y_sum**2/float(y_size))/float(y_size)
        #preloop initialization
        cdef int feature_ind, t_ind
        cdef np.ndarray[np.float64_t, ndim=2] X_transpose
        cdef np.ndarray[np.float64_t, ndim=1] feature_sorted, y_sorted
        cdef double left_size, right_size, left_sum, left_sum_sq, right_sum, right_sum_sq
        cdef double left_impurity, right_impurity, information_gain
        
        X_transpose = np.asfortranarray(X.T)
        #loop on features, searching best threshold
        for feature_ind in xrange(self.features_num):
            feature_sorted = X_transpose[feature_ind][sorted_ind[feature_ind]]
            y_sorted = y[sorted_ind[feature_ind]]
            
            left_size, right_size = 0.0, float(y_size)
            left_sum = left_sum_sq = 0.0
            right_sum = y_sum
            right_sum_sq = y_sum_sq
            
            for t_ind in xrange(y_size - 1):
                left_sum += y_sorted[t_ind]
                right_sum -= y_sorted[t_ind]
                left_sum_sq += y_sorted[t_ind]**2
                right_sum_sq -= y_sorted[t_ind]**2
                left_size += 1.0
                right_size -= 1.0
                
                if feature_sorted[t_ind] == feature_sorted[t_ind + 1]:
                    continue

                left_impurity = left_sum_sq - 1.0/left_size * left_sum**2
                right_impurity = right_sum_sq - 1.0/right_size * right_sum**2
                information_gain = y_impurity - 1.0/float(y_size) * (left_impurity + right_impurity)
                
                if information_gain > max_gain:
                    max_gain = information_gain
                    best_feature_ind = feature_ind
                    best_threshold = (feature_sorted[t_ind] + feature_sorted[t_ind + 1])/2.0
                    best_threshold_ind = t_ind
                    
        return [max_gain, best_feature_ind, best_threshold, best_threshold_ind]
    
    
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def make_split(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y,
                   np.ndarray[long, ndim=2] sorted_ind, int feature_ind, double threshold):
        cdef np.ndarray[long, ndim=2] sorted_ind_left, sorted_ind_right
        cdef np.ndarray[long, ndim=1] left, right, sorted_1d
        cdef dict index_dict_left, index_dict_right
        
        condition = X.T[feature_ind] <= threshold
        left = np.where(condition)[0]
        
        right = np.where(np.invert(condition))[0]
        index_dict_left = {}
        index_dict_right = {}
        cdef int i
        
        for i in xrange(left.shape[0]):
            index_dict_left[left[i]] = i
        for i in xrange(right.shape[0]):
            index_dict_right[right[i]] = i
            
        sorted_ind_left = np.zeros((self.features_num, left.shape[0])).astype(int)
        sorted_ind_right = np.zeros((self.features_num, right.shape[0])).astype(int)
        cdef int f_ind, idx_left, idx_right    
        
        for f_ind in xrange(self.features_num):
            sorted_1d = sorted_ind[f_ind]
            idx_left = 0
            idx_right = 0
            for i in xrange(sorted_1d.shape[0]):
                if sorted_1d[i] in index_dict_left:
                    sorted_ind_left[f_ind][idx_left] = index_dict_left[sorted_1d[i]]
                    idx_left += 1
                else:
                    sorted_ind_right[f_ind][idx_right] = index_dict_right[sorted_1d[i]]
                    idx_right += 1
        return X[left], y[left], X[right], y[right], sorted_ind_left, sorted_ind_right
    
    
    def build(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y,
              np.ndarray[long, ndim=2] sorted_ind, int depth=0):
        tree = Tree()
        tree.depth = depth
        tree.prediction = np.mean(y)
        tree.samples = y.size
        tree.mse = self.mse(y)

        cdef np.ndarray[long, ndim=2] sorted_ind_left, sorted_ind_right
        cdef np.ndarray[long, ndim=1] sorted_1d
        cdef double gain
        cdef list index_list
        cdef int i, f_ind
        
        if not self.stop_criterion(tree, y):
            tree.gain, tree.feature_ind, tree.threshold, tree.threshold_ind = self.best_split(X, y, sorted_ind)
            if tree.gain == 0.0:
                return tree
            X_left, y_left, X_right, y_right, sorted_ind_left, sorted_ind_right = self.make_split(X, y,
                                                                                                  sorted_ind, 
                                                                                                  tree.feature_ind, 
                                                                                                  tree.threshold)
            tree.left = self.build(X_left, y_left, sorted_ind_left, tree.depth+1)
            tree.right = self.build(X_right, y_right, sorted_ind_right, tree.depth+1)
        return tree
    
    
    def fit(self, np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=1] y):
        cdef int features_num
        cdef np.ndarray[long, ndim=2] sorted_ind
        
        self.features_num = X.shape[1]
        sorted_ind = self.presort(X)
        self.tree = self.build(X, y, sorted_ind)
        return
    

    def evaluate_tree(self, np.ndarray[np.float64_t, ndim=2] X, tree):
        cdef np.ndarray[np.float64_t, ndim=1] left_prediction, rigth_prediction, prediction
        
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
        
    
    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
        cdef np.ndarray[np.float64_t, ndim=1] prediction
        prediction = self.evaluate_tree(X, self.tree)
        return prediction
    
    
    def add_weights_node(self, X, y, sigmoid_output, antigradient, tree):
        if tree.left == None:
            numerator = np.sum(antigradient)
            denominator = np.sum(sigmoid_output*(1.0 - sigmoid_output))
            if abs(denominator) < 1e-150:
                tree.prediction = 0.0
            else:
                tree.prediction = numerator / denominator
            return
        left_index = X.T[tree.feature_ind] <= tree.threshold
        right_index = np.invert(left_index)
        self.add_weights_node(X[left_index], y[left_index], sigmoid_output[left_index],
                              antigradient[left_index], tree.left)
        self.add_weights_node(X[right_index], y[right_index], sigmoid_output[right_index],
                              antigradient[right_index], tree.right)
        return
    
    def add_weights(self, X, y, sigmoid_output, antigradient):
        self.add_weights_node(X, y, sigmoid_output, antigradient, self.tree)
        return
        
    
    def print_node(self, tree):
        print "DEPTH: %d" %tree.depth
        print "feature_ind: %d" %tree.feature_ind
        print "threshold: %f" %tree.threshold
        print "samples: %d" %tree.samples
        print "gain: %f" %tree.gain
        print "mse: %f" %tree.mse
        print "prediction: %f" %tree.prediction
        print "========================"
        if tree.left is None:
            return
        else:
            self.print_node(tree.left)
            self.print_node(tree.right)
        return
    
    def print_tree(self):
        self.print_node(self.tree)
        return
        


class Decision_tree(Decision_tree_, BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=10000, min_samples_split=1, min_impurity=1e-50, min_gain=0.0):
        super(Decision_tree, self).__init__(max_depth, min_samples_split, min_impurity, min_gain)
