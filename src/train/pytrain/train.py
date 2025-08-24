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
import importlib
import inspect
import sys

# Torch for datasets and training tools
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# Opacus for differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes

# Onnx for saving trained models
import onnx
from onnx2pytorch import ConvertModel

from .task_base import TaskBase

from .model_constructor import *
from .dataset_constructor import *
from .loss_constructor import *
from .eval_tools import *


class Train(TaskBase):
    """
    Args:
    config: training configuration 

    Methods:
    init: initializes the model, dataloader, optimizer, scheduler and privacy engine
    load_data: loads data from csv as data loaders
    load_model: loads model object from model config
    load_optimizer: loads model optimizer from model config
    load_loss_fn: loads loss function from config
    make_dprivate: makes model,dataloader and optimizer private
    train: trains the model
    inference: inference on the validation set
    execute: main function which includes all the above functions
    """

    def init(self, config):
        self.device = torch.device(config.get("device"))
        self.is_private = config.get("is_private", False)
        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.custom_loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None
        self.model_non_dp = None

    def load_data(self):
        dataset_config = self.config.get("dataset_config")
        all_splits = create_dataset(dataset_config, self.config.get("paths", {}).get("input_dataset_path"))

        if "train" in all_splits:
            train_dataset = all_splits["train"]
            print(f"Training samples: {len(train_dataset)}")
            self.train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size"), shuffle=True, num_workers=0)
        if "val" in all_splits:
            val_dataset = all_splits["val"]
            print(f"Validation samples: {len(val_dataset)}")
            self.val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size"), shuffle=True, num_workers=0)
        if "test" in all_splits:
            test_dataset = all_splits["test"]
            print(f"Test samples: {len(test_dataset)}")
            self.test_loader = DataLoader(test_dataset, batch_size=self.config.get("batch_size"), shuffle=True, num_workers=0)

        if self.config.get("is_private") == True:
            self.config["privacy_params"]["delta"] = 1/len(train_dataset)

        print("Dataset constructed from config")


    def load_model(self):
        model_type = self.config.get("model_type")
        if model_type == "onnx":
            onnx_model = onnx.load(self.config.get("paths", {}).get("base_model_path"))
            model = ConvertModel(onnx_model, experimental=True)

            print("Model loaded from ONNX file")
            self.model = model.to(self.device)

            if self.is_private:
                model_non_dp = onnx.load(self.config.get("paths", {}).get("base_model_path"))
                model_non_dp = ConvertModel(model_non_dp, experimental=True)
                self.model_non_dp = model_non_dp.to(self.device)
                print("Created non-private baseline model for comparison")

        elif model_type == "pytorch":
            # self.model = TorchNNModel.from_config_dict(self.config.get("model_config"))
            # self.model.load_state_dict(torch.load(self.config.get("paths", {}).get("saved_weights_path")))
            self.model = ModelFactory.load_from_dict(self.config.get("model_config"))
            self.model = self.model.to(self.device)
            print("Custom model loaded from PyTorch config")

            if self.is_private:
                self.model_non_dp = ModelFactory.load_from_dict(self.config.get("model_config"))
                self.model_non_dp = self.model_non_dp.to(self.device)
                print("Created non-private baseline model for comparison")


    def load_optimizer(self):
        optimizer_name = self.config.get("optimizer", {}).get("name", "adam")
        optimizer_params = self.config.get("optimizer", {}).get("params", {})
        optimizer_class = getattr(optim, optimizer_name)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        if self.is_private:
            self.optimizer_non_dp = optimizer_class(self.model_non_dp.parameters(), **optimizer_params)

        print(f"Optimizer {optimizer_name} loaded from config")

        if self.config.get("scheduler") is not None:
            scheduler_name = self.config.get("scheduler", {}).get("name", "cyclic")
            scheduler_params = self.config.get("scheduler", {}).get("params", {})
            scheduler_class = getattr(lr_scheduler, scheduler_name)
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
            print(f"Scheduler {scheduler_name} loaded from config")


    def load_loss_fn(self):
        if self.config.get("loss_config") is not None:
            self.custom_loss_fn = LossComposer.load_from_dict(self.config.get("loss_config"))
            print("Custom loss function loaded from config")
        else:
            # Raise an error if no loss function configuration is found
            raise ValueError("No loss function configuration found. Please provide a loss function configuration.")


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


    def train(self):
        run_val = True if self.val_loader is not None else False

        for epoch in range(self.config.get("total_epochs")):
            # set model to train mode
            self.model.train()

            train_loss = 0
            for [inputs, labels] in self.train_loader:
                # move inputs and labels to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the gradients
                self.optimizer.zero_grad()

                # forward pass
                pred = self.model(inputs)

                # compute loss
                loss = self.custom_loss_fn.calculate_loss(pred, labels)
                train_loss += loss.item()

                # backward pass
                loss.backward()

                # update weights
                self.optimizer.step()

                # update learning rate per batch - optional
                if self.scheduler is not None:
                    self.scheduler.step()

                epsilon = None
                if self.privacy_engine is not None:
                    epsilon = self.privacy_engine.get_epsilon(self.config.get("privacy_params", {}).get("delta"))
            
            # update learning rate per epoch - optional
            # self.scheduler.step()

            eps_str = f"| Epsilon: {epsilon:.4f}" if epsilon is not None else ""
            print(f"Epoch {epoch+1}/{self.config.get('total_epochs')} completed | Training Loss: {train_loss/len(self.train_loader):.4f} {eps_str}")

            if run_val:
                val_loss = 0
                with torch.no_grad():
                    for [inputs, labels] in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        pred = self.model(inputs)
                        loss = self.custom_loss_fn.calculate_loss(pred, labels)
                        val_loss += loss.item()
                print(f"Epoch {epoch+1}/{self.config.get('total_epochs')} completed | Validation Loss: {val_loss/len(self.val_loader):.4f}")

            # --- END OF TRAINING ---

            # If privacy is enabled, train a non-private replica model for comparison
            if self.is_private:
                print("\nTraining non-private replica model for comparison...")
                
                # Train non-private model
                self.model_non_dp.train()
                for epoch in range(self.config.get("total_epochs")):
                    train_loss = 0                    
                    for [inputs, labels] in self.train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer_non_dp.zero_grad()
                        pred = self.model_non_dp(inputs)
                        loss = self.custom_loss_fn.calculate_loss(pred, labels)
                        train_loss += loss.item()
                        loss.backward()
                        self.optimizer_non_dp.step()
                        
                    print(f"Non-private baseline model - Epoch {epoch+1}/{self.config.get('total_epochs')} completed | Training Loss: {train_loss/len(self.train_loader):.4f}")
                    
                    if run_val:
                        val_loss = 0
                        with torch.no_grad():
                            for [inputs, labels] in self.val_loader:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                pred = self.model_non_dp(inputs)
                                loss = self.custom_loss_fn.calculate_loss(pred, labels)
                                val_loss += loss.item()
                        print(f"Non-private baseline model - Epoch {epoch+1}/{self.config.get('total_epochs')} completed | Validation Loss: {val_loss/len(self.val_loader):.4f}")
                    

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
            in_shape = (1,) + tuple(self.train_loader.dataset[0][0].shape)
            torch.onnx.export(
                self.model,
                torch.randn(in_shape),
                output_path,
                verbose=False,
            )


    def inference(self):
        if self.test_loader is None:
            print("Test loader is not defined. Skipping inference.")
            return

        save_path = self.config.get("paths", {}).get("sample_predictions_path", None)
        metrics = parse_metrics_config(self.config.get("metrics", []))
        task_type = self.config.get("task_type", "classification")
        n_pred_samples = self.config.get("n_pred_samples", 5)

        self.model.eval()
        preds_list = []
        targets_list = []

        test_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch["input"], batch.get("target", None)

                x = x.to(self.device)
                y = y.to(self.device) if y is not None else None
                pred = self.model(x)

                loss = self.custom_loss_fn.calculate_loss(pred, y)
                test_loss += loss.item()

                preds_list.extend([p.detach().squeeze().cpu().numpy() for p in pred])
                if y is not None:
                    targets_list.extend([t.detach().squeeze().cpu().numpy() for t in y])
       
        # compute test loss
        test_loss = test_loss / len(self.test_loader)

        # compute metrics
        numeric_metrics = compute_metrics(preds_list, targets_list, test_loss, self.config)

        print(f"Evaluation Metrics: {numeric_metrics}")


    def execute(self, config):
        try:
            self.init(config)
            self.load_data()
            self.load_model()
            self.load_optimizer()
            self.load_loss_fn()
            if self.config.get("is_private") == True:
                self.make_dprivate()

            # --- START OF TRAINING ---
            self.train()
            # --- END OF TRAINING ---

            self.save_model()

            # run evaluation on test set
            self.inference()

            print("CCR Training complete!\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e