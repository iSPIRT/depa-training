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
import h5py

mnist_input_folder='/mnt/input/data/'

# Location of preprocessed MNIST dataset
mnist_output_folder='/mnt/output/preprocessed/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

trainset = torchvision.datasets.MNIST(root=mnist_input_folder, train=True, download=True, transform=transform)

# Build tensors (N, C, H, W) and labels (N,)
features = []
targets = []
for img, label in trainset:
    # img is a tensor (1, 28, 28)
    features.append(img)
    targets.append(label)

features = torch.stack(features).to(torch.float32)
targets = torch.tensor(targets, dtype=torch.int64)

# Ensure output directory exists
os.makedirs(mnist_output_folder, exist_ok=True)

# Save as HDF5 with keys 'features' and 'targets'
out_path = os.path.join(mnist_output_folder, 'mnist-dataset.h5')
with h5py.File(out_path, 'w') as f:
    f.create_dataset('features', data=features.numpy())
    f.create_dataset('targets', data=targets.numpy())

print(f"Saved MNIST dataset to {out_path} as HDF5 with keys 'features' and 'targets'.")
