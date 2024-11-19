import pickle

from dataset import load_or_cache_data, load_cifar10
from evaluate_models import evaluate_model
from models.gaussian_naive_bayes import GaussianNaiveBayes
from models.decision_tree import DecisionTree
from models.convolution_nn import ConvolutionNeuralNetwork
from models.mlp import MultiLayerPerceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import torchvision.transforms as transforms
import os
import torch
model_data_path = "./data/model_data"

if __name__ == '__main__':
    # Set the device to GPU if available (mps is used for Apple Silicon processors)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")
    # Load or cache the data
    train_feature, train_labels, test_features, test_labels = load_or_cache_data(batch_size=32, device=device)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # # Train and evaluate using my Gaussian Naive Bayes implementation
    # model = GaussianNaiveBayes()
    # test_pred = model.fit(train_feature, train_labels).predict(test_features)
    # evaluate_model("My GaussianNB model", test_labels, test_pred, classes)
    #
    # # Train and evaluate using sklearn's Gaussian Naive Bayes implementation
    # model = GaussianNB()
    # test_pred = model.fit(train_feature, train_labels).predict(test_features)
    # evaluate_model("SK GaussianNB model", test_labels, test_pred, classes)
    #
    # # Train and evaluate using my Decision Tree implementation with different depths
    # decision_tree_depths = [1, 5, 10, 15, 50]
    # for depth in decision_tree_depths:
    #     model = DecisionTree(depth, file_path=f"{model_data_path}/decision_tree_model_{depth}.pkl")
    #     test_pred = model.fit(train_feature, train_labels).predict(test_features)
    #     evaluate_model(f"My Decision Tree model with depth {depth}", test_labels, test_pred, classes)
    #
    # # Train and evaluate using sklearn's Decision Tree implementation
    # model = DecisionTreeClassifier(max_depth=20)
    # test_pred = model.fit(train_feature, train_labels).predict(test_features)
    # evaluate_model("SK Decision Tree model",test_labels, test_pred, classes)
    #
    # # Train and evaluate using my Multi Layer Perceptron implementation
    # num_epochs = 10
    # model = MultiLayerPerceptron(device=device, num_epochs=num_epochs, hidden_layer_size=512, layer_removal=True, file_path=f"{model_data_path}/bestMLP_model_{num_epochs}_removed_layer.pth")
    # test_pred = (model.fit(train_features=train_feature, train_labels=train_labels)
    #              .predict(test_features=test_features, test_labels=test_labels))
    # evaluate_model(f"My MLP model with {num_epochs} epochs and removed layer", test_labels, test_pred, classes)
    #
    # model = MultiLayerPerceptron(device=device, num_epochs=num_epochs, hidden_layer_size=512, layer_addition=True, file_path=f"{model_data_path}/bestMLP_model_{num_epochs}_additional_layer.pth")
    # test_pred = (model.fit(train_features=train_feature, train_labels=train_labels)
    #              .predict(test_features=test_features, test_labels=test_labels))
    # evaluate_model(f"My MLP model with {num_epochs} epochs and additional layer", test_labels, test_pred, classes)
    #
    #
    # hidden_layer_sizes = [256, 512, 1024]
    # # Evaluate the model with different hidden layer sizes
    # for hidden_layer_size in hidden_layer_sizes:
    #     model = MultiLayerPerceptron(device=device, num_epochs=num_epochs, hidden_layer_size=hidden_layer_size, file_path=f"{model_data_path}/bestMLP_model_{num_epochs}_{hidden_layer_size}.pth")
    #     test_pred = (model.fit(train_features=train_feature, train_labels=train_labels)
    #                  .predict(test_features=test_features, test_labels=test_labels))
    #     evaluate_model(f"My MLP model with {num_epochs} epochs and hidden layer size {hidden_layer_size}",test_labels, test_pred, classes)

    # Train and evaluate using my Convolution Neural Network implementation
    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the CIFAR-10 dataset with the new transforms
    train_loader, test_loader = load_cifar10(transform, 32)

    num_epochs = 200

    # Evaluate the model with different kernel sizes
    kernel_sizes = [2, 3, 5]
    for kernel_size in kernel_sizes:
        model = ConvolutionNeuralNetwork(device=device, num_epochs=num_epochs, kernel_size=kernel_size, file_path=f"{model_data_path}/bestCNN_model_{num_epochs}_{kernel_size}.pth")
        test_pred = model.fit(dataset=train_loader.dataset).predict(data_loader=test_loader)
        evaluate_model("My CNN model",test_labels, test_pred, classes)

    # Evaluate the model with layer addition
    # model = ConvolutionNeuralNetwork(device=device, num_epochs=num_epochs, layer_addition=True, file_path=f"{model_data_path}/bestCNN_model_{num_epochs}_additional_layer.pth")
    # test_pred = model.fit(dataset=train_loader.dataset).predict(data_loader=test_loader)
    # evaluate_model("My CNN model with additional layer", test_labels, test_pred, classes)
    #
    # # Evaluate the model with layer removal
    # model = ConvolutionNeuralNetwork(device=device, num_epochs=num_epochs, layer_removal=True, file_path=f"{model_data_path}/bestCNN_model_{num_epochs}_removed_layer.pth")
    # test_pred = model.fit(dataset=train_loader.dataset).predict(data_loader=test_loader)
    # evaluate_model("My CNN model with removed layer", test_labels, test_pred, classes)

