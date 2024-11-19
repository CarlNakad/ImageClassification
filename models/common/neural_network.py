import os
import time
from torch import optim
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def fit(self, dataset=None, train_features=None, train_labels=None):
        if os.path.exists(self.file_path):
            self.load_state_dict(torch.load(self.file_path, weights_only=True))
            return self

        self.to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        best_accuracy = 0

        if train_features is not None and train_labels is not None:
            train_features = torch.tensor(train_features, dtype=torch.float32)
            train_labels = train_labels.clone().detach().to(torch.float32)
            dataset = TensorDataset(train_features, train_labels)

        train_size = int(0.9 * len(dataset))
        validation_size = len(dataset) - train_size

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                          [train_size, validation_size])

        # Split the data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False,
                                                   num_workers=0)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, pin_memory=False,
                                                        num_workers=0)

        training_time = time.time()
        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0
            timer = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            validation_accuracy = self.evaluate(validation_loader)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Accuracy: {validation_accuracy * 100:.2f}%, Time: {time.time() - timer:.2f}s')

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                # Optionally save the model
                torch.save(self.state_dict(), self.file_path)

        print(f"Training elapsed time: {time.time() - training_time:.2f}s")
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

    def predict(self, data_loader=None, test_features=None, test_labels=None):
        self.to(self.device)
        self.eval()
        predictions = []

        if test_features is not None and test_labels is not None:
            test_features = torch.tensor(test_features, dtype=torch.float32)
            test_labels = test_labels.clone().detach().to(torch.float32)
            test_dataset = TensorDataset(test_features, test_labels)
            data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=True, num_workers=2)

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return predictions