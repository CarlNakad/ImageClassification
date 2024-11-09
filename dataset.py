from sklearn.decomposition import PCA
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def filter_dataset(dataset, total_samples):
    num_classes = 10
    class_counts = {i: 0 for i in range(num_classes)}
    indices = []
    for i, (_, label) in enumerate(dataset):
        if class_counts[label] < total_samples:
            indices.append(i)
            class_counts[label] += 1
            if sum(class_counts.values()) >= total_samples * 10:
                break
    return torch.utils.data.Subset(dataset, indices)

def load_cifar10(batch_size):
    # Resize and Normalize for ResNet-18
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full CIFAR-10 dataset
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    # Filter the datasets
    train_set = filter_dataset(full_train_set, 500)
    test_set = filter_dataset(full_test_set, 100)

    # Create DataLoaders for the filtered datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def extract_features(data_loader, model, device):
    all_features = []
    all_labels = []
    model.to(device)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            all_features.append(outputs.cpu())
            all_labels.append(labels)
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels

def load_data(batch_size, device):
    print(f"Using device: {device}")
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size)

    print("ResNet-18 model as a feature extractor...")
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
    resnet18.eval()

    print("Extracting features from train images...")
    train_features, train_labels = extract_features(train_loader, resnet18, device)
    print("Extracting features from test images...")
    test_features, test_labels = extract_features(test_loader, resnet18, device)

    print("Applying PCA to reduce dimensionality...")
    pca = PCA(n_components=50)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)

    return train_loader, test_loader, train_features, train_labels, test_features, test_labels