import onnx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the CSV data
#data = pd.read_csv('/tmp/sandbox_icmr_cowin_index_without_key_identifiers.csv')
data = pd.DataFrame(np.random.randint(0,100,size=(2119, 11)), columns=['A','B','C','D','E','F','G','H','I','J','K'])

features = data.drop(columns=["K"])
target = data["K"]

# Step 2: Preprocess the data
# Assuming your target column is named 'target'
#features = data.drop(columns=['icmr_a_icmr_test_result'])
#target = data['icmr_a_icmr_test_result']

# Split the data into training and validation sets
train_features, val_features, train_target, val_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Step 3: Create a PyTorch dataset and data loader
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

train_dataset = CustomDataset(train_features, train_target)
val_dataset = CustomDataset(val_features, val_target)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Step 4: Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 5: Choose a loss function and optimizer
model = SimpleModel(input_dim=train_features.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    #val_loss /= len(val_loader
    #print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}')

print('Training finished.')

torch.onnx.export(model, torch.randn(1, train_features.shape[1]), "/mnt/model/model.onnx", verbose=True)
print('Model saved as ONNX.')
                   
#model.save('/mnt/model/dpsgd_model')

#model.save_weights('/mnt/model/model_weights')

#model.save('/mnt/model/dpsgd_model.h5')
