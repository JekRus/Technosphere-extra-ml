import numpy as np
import threading
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


class Decision_tree_multithread(BaseEstimator, RegressorMixin):
    """My Decision tree with multithreading"""
    
    def __init__(self, max_depth=None, min_leaf_size=1, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_impurity = min_impurity
        self.tree = Tree()
    
    def presort(self, X):
        return np.argsort(X, axis=0).T
    
    def mse(self, y):
        return 1.0/float(y.size) * np.sum((y - np.mean(y))**2)
    
    def stop_criterion(self, tree, y, depth):
        if self.max_depth is not None and depth == self.max_depth:
            return True
        if y.size <= self.min_leaf_size:
            return True
        if len(np.unique(y)) == 1:
            return True
        if self.mse(y) <= self.min_impurity:
            return True
        return False
    
    def check_stop(self, node_queue, data_queue, depth):
        node_ind = 0
        while node_ind != len(node_queue):
            node = node_queue[node_ind]
            _, y, _ = data_queue[node_ind]
            if self.stop_criterion(node, y, depth):
                node.prediction = np.mean(y)
                del node_queue[node_ind]
                del data_queue[node_ind]
            else:
                node_ind += 1
        return
    
    
    def best_split_thread(self, node_queue, data_queue, f_nums, y_sum, y_sum_sq, y_size, y_impurity, idx, result):
        max_gain = [0.0] * len(node_queue)
        best_feature_ind = [0] * len(node_queue)
        best_threshold = [0] * len(node_queue)
        best_threshold_ind = [0] * len(node_queue)
        
        for f_ind in f_nums:
            max_gain_f = [0.0] * len(node_queue)
            best_threshold_f = [0] * len(node_queue)
            best_threshold_ind_f = [0] * len(node_queue)
            ## inner
            for node_ind in xrange(len(node_queue)):
                node = node_queue[node_ind]
                X, y, sorted_ind = data_queue[node_ind]
                feature_sorted = X.T[f_ind][sorted_ind[f_ind]]
                y_sorted = y[sorted_ind[f_ind]]
                left_size, right_size = 0, y_sorted.size
                left_sum = left_sum_sq = 0.0
                right_sum = y_sum[node_ind]
                right_sum_sq = y_sum_sq[node_ind]
                ### searching for best threshold
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
                    information_gain = (y_impurity[node_ind] - 1.0/float(y_sorted.size) * 
                                        (left_impurity + right_impurity))

                    if information_gain > max_gain_f[node_ind]:
                        max_gain_f[node_ind] = information_gain
                        best_threshold_f[node_ind] = (feature_sorted[t_ind] + feature_sorted[t_ind + 1])/2.0
                        best_threshold_ind_f[node_ind] = t_ind
                if max_gain_f[node_ind] > max_gain[node_ind]:
                    max_gain[node_ind] = max_gain_f[node_ind]
                    best_feature_ind[node_ind] = f_ind
                    best_threshold[node_ind] = best_threshold_f[node_ind]
                    best_threshold_ind[node_ind] = best_threshold_ind_f[node_ind]
        result[idx].extend([max_gain, best_feature_ind, best_threshold, best_threshold_ind])
        return 
    
    def best_split(self, node_queue, data_queue):
        ### initialization for every node
        max_gain = [0.0] * len(node_queue)
        best_feature_ind = [0] * len(node_queue)
        best_threshold = [0] * len(node_queue)
        best_threshold_ind = [0] * len(node_queue)
        ### precalculated summs for every node for speed up
        y_sum = np.array([np.sum(data[1]) for data in data_queue])
        y_sum_sq = np.array([np.sum(data[1]**2) for data in data_queue])
        y_size = np.array([data[1].size for data in data_queue])
        y_impurity = (y_sum_sq - y_sum**2/y_size)/y_size
        
        #miltuthreading
        threads_num = 4
        
        features_per_thread = self.features_num / threads_num
        features = [i for i in xrange(self.features_num)]
        extra_features = self.features_num % threads_num
        thread_f = [None] * threads_num
        
        for thread_idx in xrange(threads_num):
            thread_f[thread_idx] = features[thread_idx*features_per_thread:(thread_idx+1)*features_per_thread]
        thread_idx = 0
        while extra_features > 0:
            thread_f[thread_idx].append(features[-extra_features])
            thread_idx += 1
            extra_features -= 1
        
        threads = []
        result = []
        for thread_idx in xrange(threads_num):
            if len(thread_f[thread_idx]) != 0:
                result.append([])
                args = (node_queue, data_queue, thread_f[thread_idx], y_sum, y_sum_sq, y_size, y_impurity, thread_idx, result)
                threads.append(threading.Thread(target=self.best_split_thread, args=args))
            else:
                break
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        for i in xrange(len(threads)):
            thread_result = result[i]
            for node_ind in xrange(len(node_queue)):
                if thread_result[0][node_ind] > max_gain[node_ind]:
                    max_gain[node_ind] = thread_result[0][node_ind]
                    best_feature_ind[node_ind] = thread_result[1][node_ind]
                    best_threshold[node_ind] = thread_result[2][node_ind]
                    best_threshold_ind[node_ind] = thread_result[3][node_ind]
            
        
        
        return max_gain, best_feature_ind, best_threshold, best_threshold_ind
    
    def make_split(self, node, X, y, sorted_ind):
        left = np.where(X.T[node.feature_ind] <= node.threshold)[0]
        right = np.where(X.T[node.feature_ind] > node.threshold)[0]
        index_list = [None] * X.shape[0]
        for i in xrange(left.shape[0]):
            index_list[left[i]] = (i, "l")
        for i in xrange(right.shape[0]):
            index_list[right[i]] = (i, "r")

        sorted_ind_left = np.zeros((self.features_num, left.shape[0])).astype(int)
        sorted_ind_right = np.zeros((self.features_num, right.shape[0])).astype(int)

        for f_ind in xrange(self.features_num):
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
        return X[left], X[right], y[left], y[right], sorted_ind_left, sorted_ind_right
    
    
    def build(self, X, y, sorted_ind):
        tree = Tree()
        depth = 0
        self.features_num = X.shape[1]
        ### node queue: queue of nodes which we need to build
        ### data queue: X and y for every node
        node_queue = []
        data_queue = []
        node_queue.append(tree)
        data_queue.append((X, y, sorted_ind))
        
        while len(node_queue) != 0:
            ### check if some nodes are leaves
            self.check_stop(node_queue, data_queue, depth)
            ### find best split for each node
            max_gain, best_feature_ind, best_threshold, best_threshold_ind = self.best_split(node_queue, data_queue)
            nodes_num = len(node_queue)
            ### building nodes and split data
            for node_ind in xrange(nodes_num):
                node = node_queue[node_ind]
                node.feature_ind = best_feature_ind[node_ind]
                node.threshold = best_threshold[node_ind]
                node.threshold_ind = best_threshold_ind[node_ind]
                X, y, sorted_ind = data_queue[node_ind]
                X_left, X_right, y_left, y_right, ind_left, ind_right = self.make_split(node, X, y, sorted_ind)
                
                node.left = Tree()
                node.right = Tree()
                node_queue.append(node.left)
                node_queue.append(node.right)
                data_queue.append((X_left, y_left, ind_left))
                data_queue.append((X_right, y_right, ind_right))
            node_queue = node_queue[nodes_num:]
            data_queue = data_queue[nodes_num:]
            depth += 1
            
        return tree
    
    
    def fit(self, X, y):
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
