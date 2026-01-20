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

import torch 
import torch.nn as nn
import torch.nn.functional as F
import os

# Disable the new ONNX exporter and use the legacy one for compatibility
os.environ["TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK"] = "ERRORS_ONLY"

model_path="/mnt/model/"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# Define the input size for MNIST (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model using legacy exporter
# Use opset 11 for compatibility with onnx2pytorch
with torch.no_grad():
    torch.onnx.export(
        net, 
        dummy_input, 
        model_path + "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        dynamo=False  # Use legacy exporter, not the new dynamo-based one
    )