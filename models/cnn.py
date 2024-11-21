import torch.nn as nn
from models.common.neural_network import NeuralNetwork


class ConvolutionNeuralNetwork(NeuralNetwork):
    def __init__(self, device="cpu", num_epochs=10, kernel_size=3, file_path="bestCNN_model.pth", layer_addition=False, layer_removal=False):
        """
        Convolution Neural Network (CNN) with VGG11 architecture.
        The model can be trained with an additional layer or with a layer removed.

        :param device: Device to run the model on
        :param num_epochs: Number of epochs to train the model
        :param kernel_size: Kernel size for the convolutional layers
        :param file_path: File path to save the model
        :param layer_addition: Whether to add an extra layer to the model
        :param layer_removal: Whether to remove a layer from the model
        """

        super().__init__()
        self.num_epochs = num_epochs
        self.device = device
        self.file_path = file_path

        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size, padding=padding),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        if layer_addition:
            self.features.add_module("extra_conv", nn.Conv2d(512, 512, kernel_size, padding=padding))
            self.features.add_module("extra_bn", nn.BatchNorm2d(512))
            self.features.add_module("extra_relu", nn.ReLU(True))

        if layer_removal:
            feature_layers = list(self.features.children())[:-3]
            self.features = nn.Sequential(*feature_layers)

        # Original classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

        # Adjust classifier based on layer_addition or layer_removal
        if layer_addition:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 8192),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(8192, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10)
            )

        if layer_removal:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 10)
            )

    def forward(self, x):
        """
        Forward pass of the model

        :param x: Input tensor
        :return: Output tensor
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x