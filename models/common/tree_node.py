import numpy as np

class TreeNode:
    def __init__(self, features=None, threshold=None, left=None, right=None, label=None, gini=None):
        """
        Tree Node for Decision Tree

        :param features: Feature index
        :param threshold: Threshold for the feature
        :param left: Left subtree
        :param right: Right subtree
        :param label: Label of the node
        :param gini: Gini impurity of the node
        """

        self.feature = features
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gini = gini

    def calculate_gini(D):
        """
        Calculate Gini impurity
        :param D: Data
        :return: Gini impurity
        """

        labels = D[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)