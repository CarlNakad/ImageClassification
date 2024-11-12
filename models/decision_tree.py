import numpy as np
class TreeNode:
    def __init__(self, features=None, threshold=None, left=None, right=None, label=None, gini=None):
        self.feature = features
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gini = gini

    @staticmethod
    def calculate_gini(D):
        labels = D[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def majority_class(self, D):
        labels, counts = np.unique(D[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def find_best_split(self, D):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        num_features = D.shape[1] - 1

        for feature in range(num_features):
            values = np.unique(D[:, feature])
            value_count = 0
            for value in values:
                print(f"Splitting on feature {feature}/{num_features} at value {value_count}/{len(values)}")
                value_count += 1
                d_left = D[D[:, feature] < value]
                d_right = D[D[:, feature] >= value]

                if d_left.shape[0] > 0 and d_right.shape[0] > 0:
                    gini = (d_left.shape[0] * TreeNode.calculate_gini(d_left) +
                            d_right.shape[0] * TreeNode.calculate_gini(d_right)) / D.shape[0]

                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = value

        return best_feature, best_threshold

    def all_same_label(self, D):
        labels = D[:, -1]
        return np.all(labels == labels[0])

    def build_tree(self, D, depth):
        print("Depth: ", depth)
        if D.shape[0] <= 1 or depth >= self.max_depth or self.all_same_label(D):
            return TreeNode(label=self.majority_class(D))

        best_feature, best_threshold = self.find_best_split(D)
        if best_feature is None:
            return TreeNode(label=self.majority_class(D))

        d_left = D[D[:, best_feature] < best_threshold]
        d_right = D[D[:, best_feature] >= best_threshold]

        left_subtree = self.build_tree(d_left, depth + 1)
        right_subtree = self.build_tree(d_right, depth + 1)
        return TreeNode(features=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        D = np.column_stack((X, y))
        self.tree = self.build_tree(D, depth=0)

    def predict_one(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return [self.predict_one(x, self.tree) for x in X]