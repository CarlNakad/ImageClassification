import numpy as np
class GaussianNaiveBayes:
    def fit(self, train_data, classes):
        self.classes = np.unique(classes)
        n_classes = len(self.classes)
        n_features = train_data.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)

        for i, clazz in enumerate(self.classes):
            train_data_c = train_data[classes == clazz]
            self.mean[i, :] = train_data_c.mean(axis=0)
            self.var[i, :] = train_data_c.var(axis=0)
            self.prior[i] = train_data_c.shape[0] / train_data.shape[0]
        return self

    def gaussian_pdf(self, X, mean, var):
        var = var + 1e-9
        return np.exp(-0.5 * (X - mean) ** 2 / (var)) / (np.sqrt(2 * np.pi * var))
    
    def predict(self, train_data):
        n_features = train_data.shape[0]
        n_classes = len(self.classes)

        posteriors = np.zeros((n_features, n_classes))

        for i in range(n_classes):
            likelihood = self.gaussian_pdf(train_data, self.mean[i], self.var[i])
            posterior = np.sum(np.log(likelihood), axis=1) + np.log(self.prior[i])
            posteriors[:, i] = posterior

        return self.classes[np.argmax(posteriors, axis=1)]
