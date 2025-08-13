# Copyright (c) 2025 DEPA Foundation
#
# Licensed under CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
# 
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import os
import json

# Create a PyTorch dataset and data loader
class CustomDataset(Dataset):
    def __init__(self, data_dir, target_variable, transform=None, augment=False):
        self.data_dir = data_dir
        self.target_variable = target_variable
        
        data = pd.read_csv(self.data_dir)
        features = data.drop(columns=[target_variable])
        target = data[target_variable]

        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def update_features(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]



# # Create a PyTorch dataset and data loader
# class CustomDataset(Dataset):
#     def __init__(self, features, target):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.target = torch.tensor(target.values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.target[idx]


class BaseModel(nn.Module):
    """Binary classification neural network model."""
    
    def __init__(self, input_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def custom_loss_fn(outputs, labels):
    criterion = nn.BCELoss()
    return criterion(outputs, labels.unsqueeze(1))


def custom_inference_fn(model, val_loader, device, config):
    """
    Custom inference function with additional metrics
    """
    save_path = config.get("paths", {}).get("sample_predictions_path")

    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        # for [inputs, labels] in tqdm(val_loader, desc="Running inference"):
        for [inputs, labels] in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)
            
            # Calculate loss using the same loss function
            loss = custom_loss_fn(pred, labels)
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate additional metrics
    avg_loss = total_loss / len(val_loader)
    
    # Convert to binary predictions if needed
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
    binary_labels = (np.array(all_labels) > 0.5).astype(int)
    
    # Calculate classification metrics
    accuracy = accuracy_score(binary_labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_labels, binary_predictions, average='binary')
    
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