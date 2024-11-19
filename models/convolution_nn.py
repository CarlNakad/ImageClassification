import torch.nn as nn
from models.common.neural_network import NeuralNetwork


class ConvolutionNeuralNetwork(NeuralNetwork):
    def __init__(self, device="cpu", num_epochs=10, kernel_size=3, file_path="bestCNN_model.pth"):
        super().__init__()
        self.num_epochs = num_epochs
        self.device = device
        self.file_path = file_path

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x