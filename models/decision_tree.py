import numpy as np
class TreeNode:
    def __init__(self, features=None, threshold=None, left=None, right=None, label=None, gini=None):
        self.feature = features
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gini = gini

    def calculate_gini(D, features):
        return 1 - sum(len(features)/len(D) ** 2)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def majority_class(D):
        print(D)
        return 
    
    def findBestSplit(D):
        return
    
    def buildTree(self, D, k):
        if len(D) <= k:
            return TreeNode(label=self.majority_class(D))

        best_feature, best_threshold = self.findBestSplit(D)
        
        D_left = [x for x in D if x[best_feature] < best_threshold]
        D_right = [x for x in D if x[best_feature] >= best_threshold]

        left_subtree = self.buildTree(D_left, k)
        right_subtree = self.buildTree(D_right, k)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def fit(self, X, y):
        D = [list(x) + [label] for x, label in zip(X, y)]
        self.buildTree(D, k=5)

    def predict(self, X):
        return self.model.predict(X)
