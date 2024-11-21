import torch.nn as nn
import torch.nn.functional as F

from models.common.neural_network import NeuralNetwork


class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self, device="cpu", num_epochs=10, hidden_layer_size=512, file_path="bestMLP_model.pth", layer_addition=False, layer_removal=False):
        """
        Multi-Layer Perceptron (MLP) with 2 hidden layers.
        The model can be trained with an additional layer or with a layer removed.

        :param device: Device to run the model on
        :param num_epochs: Number of epochs to train the model
        :param hidden_layer_size: Size of the hidden layer
        :param file_path: File path to save the model
        :param layer_addition: Whether to add an extra layer to the model
        :param layer_removal: Whether to remove a layer from the model
        """
        super().__init__()
        self.num_epochs = num_epochs
        self.device = device
        self.file_path = file_path
        self.layer_addition = layer_addition
        self.layer_removal = layer_removal

        self.fc1 = nn.Linear(50, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.B1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc_additional1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 10)

    def forward(self, x):
        """
        Forward pass of the model

        :param x: Input tensor
        :return: Output tensor
        """
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.layer_removal:
            x = F.relu(self.B1(self.fc2(x)))
        if self.layer_addition:
            x = F.relu(self.fc_additional1(x))
        return self.fc3(x)