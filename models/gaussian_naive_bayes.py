import numpy as np
class GaussianNaiveBayes:
    def __init__(self):
        """
        Gaussian Naive Bayes Classifier
        """
        self.classes = None
        self.mean = None
        self.var = None
        self.prior = None

    def fit(self, train_data, classes):
        """
        Fit the model

        :param train_data: Training data
        :param classes: Labels
        :return: Trained model
        """
        self.classes = np.unique(classes)
        n_classes = len(self.classes)
        n_features = train_data.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)

        # Calculate mean, variance and prior for each class
        for i, clazz in enumerate(self.classes):
            train_data_c = train_data[classes == clazz]
            self.mean[i, :] = train_data_c.mean(axis=0)
            self.var[i, :] = train_data_c.var(axis=0)
            self.prior[i] = train_data_c.shape[0] / train_data.shape[0]
        return self

    def gaussian_pdf(self, X, mean, var):
        """
        Calculate Gaussian PDF

        :param X: Data
        :param mean: Mean
        :param var: Variance
        :return: Gaussian PDF
        """

        # Add a small number to avoid division by zero
        var = var + 1e-9
        return np.exp(-0.5 * (X - mean) ** 2 / (var)) / (np.sqrt(2 * np.pi * var))
    
    def predict(self, train_data):
        """
        Predict the class

        :param train_data: Data to predict
        :return: Predicted classes
        """
        n_features = train_data.shape[0]
        n_classes = len(self.classes)

        posteriors = np.zeros((n_features, n_classes))

        # Calculate the posterior for each class
        for i in range(n_classes):
            # Calculate likelihood
            likelihood = self.gaussian_pdf(train_data, self.mean[i], self.var[i])
            # Calculate posterior
            posterior = np.sum(np.log(likelihood), axis=1) + np.log(self.prior[i])
            # Store the posterior
            posteriors[:, i] = posterior

        # Return the class with the highest posterior
        return self.classes[np.argmax(posteriors, axis=1)]
