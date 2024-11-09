import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import TensorDataset


class MultiLayerPerceptron(nn.Module):
    def __init__(self, device="cpu", num_epochs=10):
        super().__init__()
        self.num_epochs = num_epochs
        self.device = device

        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 512)
        self.B1 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.B1(self.fc2(x)))
        return self.fc3(x)

    def fit(self, train_features, train_labels):
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        best_accuracy = 0

        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)

        dataset = TensorDataset(train_features, train_labels)
        train_size = int(0.9 * len(train_features))
        validation_size = len(train_features) - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                          [train_size, validation_size])


        # Split the data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                                   num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, pin_memory=True,
                                                        num_workers=2)

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0
            load_size = len(train_loader)
            instance_tracker = 1

            for images, labels in train_loader:
                print(f"Training {instance_tracker}/{load_size}")
                instance_tracker += 1
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
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
                torch.save(self.state_dict(), 'bestMLP_model.pth')

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

    def predict(self, test_features, test_labels):
        self.to(self.device)
        self.eval()
        predictions = []

        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_labels = test_labels.clone().detach().to(torch.float32)
        dataset = TensorDataset(test_features, test_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2)

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                inputs = inputs.view(inputs.size(0), 1, 5, 10)  # Adjust the dimensions based on your input size
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return predictions