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

import os
import random
import sys
import importlib.util # For dynamic imports
import inspect

# Torch for datasets and training tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # For progress bar

# Opacus for differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes

# Onnx for saving trained models
import onnx
from onnx2pytorch import ConvertModel

# Scikit-Learn for standardization
from sklearn.preprocessing import StandardScaler

from .task_base import TaskBase


class Train(TaskBase):
    """
    Args:
    config: training configuration 

    Methods:
    init: initializes the model, dataloader, optimizer, scheduler and privacy engine
    dynamic_import_classes: dynamically imports helper classes from the helper_scripts_path
    load_data: loads data from csv as data loaders
    load_model: loads model object from model config
    load_optimizer: loads model optimizer from model config
    make_dprivate: makes model,dataloader and optimizer private
    loss_fn: loss function for training
    train: trains the model
    inference: inference on the validation set
    execute: main function which includes all the above functions
    """

    def init(self, config):
        self.device = torch.device(config.get("device"))

        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None
        self.dynamic_import_classes()


    def dynamic_import_classes(self):
        module_path = self.config.get("paths", {}).get("helper_scripts_path")  # e.g., "/mnt/remote/model"

        if not os.path.isdir(module_path):
            raise FileNotFoundError(f"Directory not found: {module_path}")

        sys.path.append(module_path)

        for filename in os.listdir(module_path):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = os.path.splitext(filename)[0]
                file_path = os.path.join(module_path, filename)

                # Load module from file
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Attach all public attributes to self
                for name, obj in inspect.getmembers(module):
                    if not name.startswith("_"):  # skip private vars
                        setattr(self, name, obj)

                print(f"Loaded helper module {module_name} from {module_path}")


    def load_data(self):
        standardize_features = self.config.get("standardize_features")
        self.dataset = self.CustomDataset(self.config.get("paths", {}).get("input_dataset_path"), target_variable=self.config.get("target_variable"), augment=False)

        # Split the dataset into train and validation sets
        train_size = int(self.config.get("train_test_split") * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if standardize_features:
            # standard scale the features
            scaler = StandardScaler()
            # Access the original dataset's features through the subset indices
            train_features = scaler.fit_transform(self.dataset.features[train_dataset.indices])
            val_features = scaler.transform(self.dataset.features[val_dataset.indices])
            
            # Update the original dataset with standardized features
            self.dataset.update_features(scaler.transform(self.dataset.features))
            
            # Recreate the subsets with the updated dataset
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        if self.config.get("is_private") == True:
            self.config["privacy_params"]["delta"] = 1/train_size

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size"), shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size"), shuffle=True, num_workers=0)


    def load_model(self):
        model_type = self.config.get("model_type")
        if model_type == "onnx":
            onnx_model = onnx.load(self.config.get("paths", {}).get("base_model_path"))
            model = ConvertModel(onnx_model, experimental=True)

            print("Model loaded from ONNX")
            self.model = model.to(self.device)

        elif model_type == "pytorch":
            self.model = self.Base_Model(**self.config.get("model_params")).to(self.device)
            self.model.load_state_dict(torch.load(self.config.get("paths", {}).get("saved_weights_path")))

            print("Model loaded from PyTorch")
            self.model = self.model.to(self.device)


    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.config.get("learning_rate"))
        self.scheduler = CyclicLR(self.optimizer, base_lr=self.config.get("learning_rate"), max_lr=self.config.get("max_lr"), cycle_momentum=False)


    def make_dprivate(self):
        self.privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config.get("total_epochs"),
            target_delta=self.config.get("privacy_params", {}).get("delta"),  # Privacy budget
            target_epsilon=self.config.get("privacy_params", {}).get("epsilon_threshold"),  # Probability of privacy breach
            max_grad_norm=self.config.get("privacy_params", {}).get("max_grad_norm"), # threshold for clipping the norm of per-sample gradients
            batch_first=True
        )

    def loss_fn(self, outputs, labels):
        if self.custom_loss_fn is not None:
            return self.custom_loss_fn(outputs, labels)
        else:
            # Default loss function
            criterion = nn.MSELoss()
            return criterion(outputs, labels.unsqueeze(1))


    def train(self):
        # set model to train mode
        self.model = self.model.train()
        
        for epoch in range(self.config.get("total_epochs")):
            # initialize progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.get('total_epochs')}")
    
            # for [inputs, labels] in progress_bar:
            for [inputs, labels] in self.train_loader:
                # move inputs and labels to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the gradients
                self.optimizer.zero_grad()

                # forward pass
                pred = self.model(inputs)

                # compute loss
                loss = self.loss_fn(pred, labels)

                # backward pass
                loss.backward()

                # update weights
                self.optimizer.step()

                # update learning rate per batch - optional
                self.scheduler.step()

                epsilon = None
                if self.privacy_engine is not None:
                    epsilon = self.privacy_engine.get_epsilon(self.config.get("privacy_params", {}).get("delta"))

                # progress_bar.set_postfix({
                #     "Loss": f"{loss.item():.4f}",
                #     "Epsilon": f"{epsilon:.4f}" if epsilon is not None else "N/A"
                # })
            
            eps_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
            print(f"Epoch {epoch+1}/{self.config.get('total_epochs')} completed | Loss: {loss.item():.4f} | Epsilon: {eps_str}")

            
            # update learning rate per epoch - optional
            # self.scheduler.step()

    def save_model(self):

        # If using differential privacy, extract the underlying model from GradSampleModule
        if isinstance(self.model, opacus.grad_sample.GradSampleModule):
            self.model = self.model._module

        # set model to eval mode
        self.model.eval()

        # save the model
        if self.config.get("model_type") == "pytorch":
            output_path = os.path.join(self.config.get("paths", {}).get("trained_model_output_path"), "trained_model.pth")
            print("Saving trained model to " + output_path)
            torch.save(self.model.state_dict(), output_path)

        elif self.config.get("model_type") == "onnx":
            output_path = os.path.join(self.config.get("paths", {}).get("trained_model_output_path"), "trained_model.onnx")
            print("Saving trained model to " + output_path)
            in_shape = (1,) + tuple(self.dataset[0][0].shape)
            torch.onnx.export(
                self.model,
                torch.randn(in_shape),
                output_path,
                verbose=False,
            )


    def inference(self):
        if self.custom_inference_fn is not None:
            return self.custom_inference_fn(self.model, self.val_loader, self.device, self.config)
        else:
            print("Custom inference function not found. Please implement a custom inference function.")


    def execute(self, config):
        try:
            self.init(config)
            self.load_data()
            self.load_model()
            self.load_optimizer()
            if self.config.get("is_private") == True:
                self.make_dprivate()

            # --- START OF TRAINING ---
            self.train()
            # --- END OF TRAINING ---

            self.save_model()

            # for testing purposes, save some validation set predictions
            self.inference()

            print("CCR Training complete!\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e