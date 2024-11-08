import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, device="cpu", num_epochs=10):
        super().__init__()
        self.num_epochs = num_epochs
        self.device = device

        self.layer1=nn.Conv2d(3,64,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(64)
        self.layer2 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(128)
        self.max_pool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(256)
        self.layer4 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(256)
        self.layer5 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(512)
        self.layer6 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(512)
        self.layer7 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.B7 = nn.BatchNorm2d(512)
        self.layer8 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.B8 = nn.BatchNorm2d(512)
        # self.fc1 = nn.Linear(512, 4096)
        # self.D1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.D2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.B1(self.max_pool(F.leaky_relu(self.layer1(x))))
        x = self.B2(self.max_pool(F.leaky_relu(self.layer2(x))))
        x = self.B3(F.leaky_relu(self.layer3(x)))
        x = self.B4(self.max_pool(F.leaky_relu(self.layer4(x))))
        x = self.B5((F.leaky_relu(self.layer5(x))))
        x = self.B6(self.max_pool(F.leaky_relu(self.layer6(x))))
        x = self.B7((F.leaky_relu(self.layer7(x))))
        x = self.B8(self.max_pool(F.leaky_relu(self.layer8(x))))
        # x = x.view(x.size(0), -1)
        # x = self.D1(F.relu(self.fc1(x)))
        # x = self.D2(F.relu(self.fc2(x)))

        # return self.fc3(x)
        return x.view(x.size(0), -1)

    def fit(self, train_loader):
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        best_accuracy = 0

        # Create train-validation split (90-10)
        train_size = int(0.9 * len(train_loader.dataset))
        validation_size = len(train_loader.dataset) - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(train_loader.dataset,
                                                                          [train_size, validation_size])

        # Split the data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                                   num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, pin_memory=True,
                                                        num_workers=4)

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0

            load_size = len(train_loader)
            instance_tracker = 0
            # Training phase
            for instances, labels in train_loader:
                print(f"Training {instance_tracker}/{load_size}")
                instance_tracker += 1
                instances, labels = instances.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(instances)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            validation_accuracy = self.evaluate(validation_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {validation_accuracy*100:.2f}%')

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                # Optionally save the model
                # torch.save(self.state_dict(), 'best_model.pth')

        return self

    def evaluate(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for instances, labels in data_loader:
                instances, labels = instances.to(self.device), labels.to(self.device)
                outputs = self(instances)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def predict(self, X):
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
            return predicted