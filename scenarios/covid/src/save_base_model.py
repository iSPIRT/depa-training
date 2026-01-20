# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and 
# related or neighboring rights to this work. 
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
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

import onnx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    """PyTorch dataset for feature-target pairs."""
    
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


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


def main():
    # Load and preprocess data
    data = pd.DataFrame(np.random.randint(0, 100, size=(2119, 10)), 
                       columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    
    features = data.drop(columns=["J"])
    target = (data["J"] > 50).astype(float)  # Binary classification

    # Split data
    train_features, val_features, train_target, val_target = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_features, train_target)
    val_dataset = CustomDataset(val_features, val_target)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = BaseModel(input_dim=train_features.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                targets = targets.unsqueeze(1)
                val_loss += criterion(outputs, targets).item()
                
                predicted = (outputs >= 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')

    print('Training finished.')

    # Save model using legacy ONNX exporter for compatibility
    with torch.no_grad():
        torch.onnx.export(
            model, 
            torch.randn(1, train_features.shape[1]), 
            "/mnt/model/model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            dynamo=False  # Use legacy exporter, not the new dynamo-based one
        )
    print('Model saved as ONNX.')


if __name__ == "__main__":
    main()
