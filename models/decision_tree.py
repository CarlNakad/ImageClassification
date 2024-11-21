import os
import pickle
import numpy as np
from models.common.tree_node import TreeNode

class DecisionTree:
    def __init__(self, max_depth=None, file_path=None):
        """
        Decision Tree Classifier

        :param max_depth: Maximum depth of the tree
        :param file_path: File path to save the model
        """

        self.max_depth = max_depth
        self.tree = None
        self.file_path = file_path

    def majority_class(self, D):
        """
        Find the majority class in the dataset

        :param D: Data
        :return: Majority class
        """

        labels, counts = np.unique(D[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def find_best_split(self, D):
        """
        Find the best split for the dataset

        :param D: Data
        :return: Best feature and threshold
        """
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        num_features = D.shape[1] - 1

        # Iterate over all features
        for feature in range(num_features):
            values = np.unique(D[:, feature])
            for value in values:
                # Split the dataset depending on the feature and threshold
                d_left = D[D[:, feature] < value]
                d_right = D[D[:, feature] >= value]

                # Calculate the Gini impurity
                if d_left.shape[0] > 0 and d_right.shape[0] > 0:
                    gini = (d_left.shape[0] * TreeNode.calculate_gini(d_left) +
                            d_right.shape[0] * TreeNode.calculate_gini(d_right)) / D.shape[0]

                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = value

        return best_feature, best_threshold

    def all_same_label(self, D):
        """
        Check if all labels are the same

        :param D: Data
        :return: True if all labels are the same, False otherwise
        """

        labels = D[:, -1]
        return np.all(labels == labels[0])

    def build_tree(self, D, depth):
        """
        Build the decision tree

        :param D: Data
        :param depth: Depth of the tree
        :return: Root node of the tree
        """

        # If the dataset is empty or the maximum depth is reached or all labels are the same, return a leaf node
        if D.shape[0] <= 1 or depth >= self.max_depth or self.all_same_label(D):
            return TreeNode(label=self.majority_class(D))


        best_feature, best_threshold = self.find_best_split(D)
        # If no split is found, return a leaf node
        if best_feature is None:
            return TreeNode(label=self.majority_class(D))

        # Split the dataset using the best feature and threshold
        d_left = D[D[:, best_feature] < best_threshold]
        d_right = D[D[:, best_feature] >= best_threshold]

        # Recursively build the left and right subtree
        left_subtree = self.build_tree(d_left, depth + 1)
        right_subtree = self.build_tree(d_right, depth + 1)
        return TreeNode(features=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        """
        Fit the model

        :param X: Features
        :param y: Labels
        :return: Trained model
        """

        # Load the model if it exists
        # otherwise train the model
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as file:
                self.tree = pickle.load(file).tree
        else:

            D = np.column_stack((X, y))
            self.tree = self.build_tree(D, depth=0)

            # Save the model
            with open(self.file_path, "wb") as file:
                pickle.dump(self, file)

        return self

    def predict_one(self, x, node):
        """
        Predict a single instance

        :param x: Instance
        :param node: Tree node
        :return: Predicted label
        """

        # If the node is a leaf node, return the label
        if node.label is not None:
            return node.label

        # Recursively traverse the tree
        if x[node.feature] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        """
        Predict the labels

        :param X: Instances
        :return: Predicted labels
        """
        return [self.predict_one(x, self.tree) for x in X]