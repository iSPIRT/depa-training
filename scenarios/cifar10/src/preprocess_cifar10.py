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

import os
import torch
import torchvision
import torchvision.transforms as transforms
from safetensors.torch import save_file as st_save

cifar10_input_folder='/mnt/input/data/'

# Location of preprocessed CIFAR-10 dataset
cifar10_output_folder='/mnt/output/preprocessed/'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])  # CIFAR-10 mean and std

trainset = torchvision.datasets.CIFAR10(root=cifar10_input_folder, train=True, download=True, transform=transform)

# Build tensors (N, C, H, W) and labels (N,)
features = []
targets = []
for img, label in trainset:
    features.append(img)
    targets.append(label)

features = torch.stack(features).to(torch.float32)
targets = torch.tensor(targets, dtype=torch.int64)

# Ensure output directory exists
os.makedirs(cifar10_output_folder, exist_ok=True)

# Save as SafeTensors with keys 'features' and 'targets'
out_path = os.path.join(cifar10_output_folder, 'cifar10-dataset.safetensors')
st_save({'features': features, 'targets': targets}, out_path)

print(f"Saved CIFAR-10 dataset to {out_path} as SafeTensors with keys 'features' and 'targets'.")
