# 2025, DEPA Foundation
#
# Licensed TBD
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Key references / Attributions: https://depa.world/training/reference-implementation
# Key frameworks used : DEPA, CCR, Opacus, PyTorch, Scikit-Learn, ONNX

import os
import random
import sys
import importlib.util # For dynamic imports
import inspect

import numpy as np
import matplotlib.pyplot as plt # for saving sample predictions
import monai # for medical imaging loss functions

# Torch for datasets and training optimizers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

# Opacus for differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes

# Onnx for saving trained models
import onnx
from onnx2pytorch import ConvertModel

from .task_base import TaskBase


class PrivateTrainVision(TaskBase):
    """
    Args:
    config:training configuration 

    Methods:
    load_data: loads data from image folders as data loaders
    load_model: loads model
    load_optimizer: loads model optimizer and scheduler
    make_dprivate: wraps model, dataloader and optimizer with DP hooks
    loss_fn: loss function for training
    train: trains the model
    execute_model: mega function which includes all the above functions
    """

    def init(self, config):
        self.device = torch.device(config["DEVICE"])

        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None
        self.dynamic_import_classes(config)

    def dynamic_import_classes(self, config):
        module_path = config["helper_scripts_path"]  # e.g., "/mnt/remote/model"

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
        dataset = self.CustomDataset(self.config["input_dataset_path"], augment=False)

        train_ratio = 1 - self.config["test_train_split"]
        n_samples = len(dataset)
        print(f"Total samples: {n_samples}")
        train_size = int(train_ratio*n_samples)

        self.config['delta'] = 1/train_size

        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, n_samples))
            
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)#, collate_fn=lambda batch: [x for x in batch if x[0] is not None and x[1] is not None])
        self.val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)#, collate_fn=lambda batch: [x for x in batch if x[0] is not None and x[1] is not None])

    
    def load_model(self):
        ## Option 1 - load model state dict. Requires the model architecture to be defined.
        self.model = self.Base_Model(in_ch=1, out_ch=1, base_ch=8, final_act='sigmoid').to(self.config['DEVICE'])
        self.model.load_state_dict(torch.load(self.config["saved_model_path"]))

        print("Model loaded")

        # ## Option 2 - load entire model using TorchScript
        # self.model = torch.jit.load(self.config["saved_model_path"])
        
        # # Option 3 - load ONNX model and convert to PyTorch
        # onnx_model = onnx.load(self.config["saved_model_path"])
        # model = ConvertModel(onnx_model, experimental=True)
        # # print(self.model)
        
        self.model = self.model.to(self.device)
    
    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.config['LEARNING_RATE'])
        self.scheduler = CyclicLR(self.optimizer, base_lr=self.config['LEARNING_RATE'], max_lr=self.config['MAX_LR'], cycle_momentum=False)
    
    def make_dprivate(self):
        self.privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed

        # for name, module in self.model.named_modules():
        #     if "Norm" in module.__class__.__name__:
        #         print(f"{name}: {module}")

                # raise ValueError("Norm layers are not supported in this model. Please remove them before training.")

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config['total_epochs'],
            target_delta=self.config['delta'],  # Privacy budget
            target_epsilon=self.config['epsilon_threshold'],  # Probability of privacy breach
            max_grad_norm=self.config['max_grad_norm'], # threshold for clipping the norm of per-sample gradients
        )

    def loss_fn(self, pred, mask):
        l1_loss = nn.L1Loss(reduction='mean')
        mse_loss = nn.MSELoss(reduction='mean')
        dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        return dice_loss(pred, mask) + 2 * l1_loss(pred, mask)


    def train(self):
        self.model = self.model.train()
        for epoch in range(self.config['total_epochs']):
            for [image, mask] in tqdm(self.train_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()

                # print(image.shape)
                
                pred = self.model(image)
                
                loss = self.loss_fn(pred, mask)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
            print(f"Epoch [{epoch+1}/{self.config['total_epochs']}], Loss: {loss.item():.4f}")

        # Extract the underlying model from GradSampleModule
        if isinstance(self.model, opacus.grad_sample.GradSampleModule):
            self.model = self.model._module

        output_path = os.path.join(self.config["trained_model_output_path"], "trained_model.pth")
        # print("Writing training model to " + output_path)
        # torch.onnx.export(
        #     self.model.to('cpu'),
        #     self.val_loader[0][0].to('cpu'),                      # model input (or a tuple for multiple inputs)
        #     output_path,                # where to save the model
        #     input_names=["image"],            # input tensor names
        #     output_names=["mask"],          # output tensor names
        #     dynamic_axes={"image": {0: "batch_size"}, "mask": {0: "batch_size"}},  # optional
        #     export_params=True,               # store the trained parameter weights
        #     verbose=True,                  # print a human readable representation of the graph
        # )

        print("Writing training model to " + output_path)
        torch.save(self.model.state_dict(), output_path)

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            i=1
            for [image, mask] in tqdm(self.val_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(image)
                loss = self.loss_fn(pred, mask)
                print(f"Validation Loss: {loss.item():.4f}")
                # Save the prediction and mask
                pred = pred[0].cpu().squeeze().numpy() > 0.1
                image = image[0].cpu().squeeze().numpy()
                mask = mask[0].cpu().squeeze().numpy() > 0.1
                plt.imsave(os.path.join(self.config["sample_predictions_path"], f"pred_{i}.png"), pred, cmap='gray')
                plt.imsave(os.path.join(self.config["sample_predictions_path"], f"image_{i}.png"), image, cmap='gray')
                plt.imsave(os.path.join(self.config["sample_predictions_path"], f"mask_{i}.png"), mask, cmap='gray')

                print(f"Sample predictions saved to {self.config['sample_predictions_path']}")
                i += 1
                if i > 5:
                    break

    def execute(self, config):
        try:
            # --- START OF TRAINING ---
            self.init(config)
            self.load_data()
            self.load_model()
            self.load_optimizer()
            self.make_dprivate()  # Differential privacy is not necessary for this task, but can be enabled if needed.
            self.train()
            print("Training complete!")
            # --- END OF TRAINING ---

            # for testing purposes, save some predictions
            self.inference()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
