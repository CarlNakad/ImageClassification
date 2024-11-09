import dataset
from models.gaussian_naive_bayes import GaussianNaiveBayes
from models.decision_tree import DecisionTree
from models.convolution_nn import ConvolutionNeuralNetwork
from models.mlp import MultiLayerPerceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import os
import torch

def load_or_cache_data(cache_path="./data/cached_data.pkl", batch_size=128, device=torch.device("cpu")):
    # Check if cached data exists
    if os.path.exists(cache_path):
        print("Loading data from cache...")
        with open(cache_path, "rb") as f:
            train_loader, test_loader, train_feature, train_labels, test_features, test_labels = pickle.load(f)
    else:
        print("Loading data from dataset and caching it...")
        # Load data if cache doesn't exist
        train_loader, test_loader, train_feature, train_labels, test_features, test_labels = dataset.load_data(batch_size=batch_size, device=device)
        # Cache the data for future use
        with open(cache_path, "wb") as f:
            pickle.dump((train_loader, test_loader, train_feature, train_labels, test_features, test_labels), f)
    
    return train_loader, test_loader, train_feature, train_labels, test_features, test_labels

if __name__ == '__main__':
    # Load or cache the data
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")
    train_loader, test_loader, train_feature, train_labels, test_features, test_labels = load_or_cache_data(batch_size=32, device=device)
    
    # Train and evaluate using my Gaussian Naive Bayes implementation
    model = GaussianNaiveBayes()
    test_pred = model.fit(train_feature, train_labels).predict(test_features)
    accuracy = accuracy_score(test_labels, test_pred)
    print(f"My implementation accuracy: {accuracy * 100:.2f}%")

    # Train and evaluate using sklearn's Gaussian Naive Bayes implementation
    model = GaussianNB()
    test_pred = model.fit(train_feature, train_labels).predict(test_features)
    accuracy = accuracy_score(test_labels, test_pred)
    print(f"SK implementation accuracy: {accuracy * 100:.2f}%")

    # model = DecisionTree(50)
    # test_pred = model.fit(train_feature, train_labels)

    model = MultiLayerPerceptron(device=device, num_epochs=10)
    if os.path.exists('bestMLP_model.pth'):
        model.load_state_dict(torch.load('bestMLP_model.pth', weights_only=True))
    else:
        model.fit(train_feature, train_labels)
    test_pred = model.predict(test_features, test_labels)

    accuracy = accuracy_score(test_labels, test_pred)
    print(f"SK implementation accuracy: {accuracy * 100:.2f}%")

    model = ConvolutionNeuralNetwork(device=device, num_epochs=10)
    if os.path.exists('bestCNN_model.pth'):
        model.load_state_dict(torch.load('bestCNN_model.pth', weights_only=True))
    else:
        model.fit(train_feature, train_labels)
    test_pred = model.predict(test_features, test_labels)

    accuracy = accuracy_score(test_labels, test_pred)
    print(f"SK implementation accuracy: {accuracy * 100:.2f}%")