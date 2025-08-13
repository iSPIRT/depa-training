import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torchvision
import torchvision.transforms as transforms
import os
import json


class CustomDataset(Dataset):
    
    def __init__(self, data_dir, target_variable=None,transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.target_variable = target_variable
        
        # Load the dataset from .pth file
        self.dataset = torch.load(data_dir, weights_only=False)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class BaseModel(nn.Module):
    """Convolutional Neural Network for MNIST/CIFAR10 classification."""
    
    def __init__(self):
        super().__init__()
        # First convolutional layer: 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for CIFAR10

    def forward(self, x):
        # First conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def custom_loss_fn(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def custom_inference_fn(model, val_loader, device, config):
    save_path = config.get("paths", {}).get("sample_predictions_path")
    
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        # for inputs, labels in tqdm(val_loader, desc="Running inference"):
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Calculate loss
            loss = custom_loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions (argmax for classification)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    metrics_dict = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # save as json
    metrics_fname = os.path.join(save_path, "validation_metrics.json")
    with open(metrics_fname, "w") as f:
        json.dump(metrics_dict, f)
    
    print(f"Validation Metrics: {metrics_dict}")
    return metrics_dict