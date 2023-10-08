# 2023, The DEPA CCR DP Training Reference Implementation
# authors shyam@ispirt.in, sridhar.avs@ispirt.in
#
# Licensed TBD
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Key references / Attributions: https://depa.world/training/reference-implementation
# Key frameworks used : DEPA CCR,Opacus, PyTorch,ONNX, onnx2pytorch

#torch related imports
from typing import Optional
import torch
from torchvision import datasets, transforms
#from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#sklearn,pandas,numpy related imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#opacus related imports
from opacus.accountants import create_accountant
from opacus import PrivacyEngine

#onnx related imports
import onnx
from onnx2pytorch import ConvertModel

#other imports
import os
import json
import argparse
from pathlib import Path

#debug_poc=True

#loading ccr_context
#with open(ccr_input_data, 'r') as json_file:
    #ccr_context = json.load(json_file)

#loading model config
#with open(model_config, 'r') as json_file:
   # model_config = json.load(json_file)

#loading logger
#with open(logger, 'r') as json_file:
   # ccr_tracking_object = json.load(json_file)

logger={
    "epochs_per_report":1,
    "metrics":["tdp_config","tdc_config","model_architecture","model_hyperparameters","model_config","accuracy","precision","recall"],
    "ccr_pbt_logger_file":"/mnt/remote/output/ccr_depa_trg_model_logger.json",
}

def compute_delta(ccr_context):
  return 1/ccr_context["sample_size"]

class CustomDataset(Dataset):
    """
    Class to convert dataset columns to tensors
    """
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
    
class ccr_model():
          """
          Args:
          model_config:model configuration built from CCR_JSON

          Methods:
          load_data:loads data from csv as data loaders
          load_model_object:loads model object from model config
          load_model_optimizer:loads model optimizer from model config
          make_dprivate:make model,dataloader and optimizer private
          execute_model:mega function which includes all the above functions

          """
          def __init__(self,model_config):
            self.model_config=model_config

          def load_data(self):
            #path from config
            data = pd.read_csv(self.model_config["input_dataset_path"])
            features = data.drop(columns=[self.model_config["target_variable"]])
            target = data[self.model_config["target_variable"]]
            train_features, val_features, train_target, val_target = train_test_split(features, target, test_size=self.model_config["test_train_split"], random_state=42)
            scaler = StandardScaler()
            self.train_features = scaler.fit_transform(train_features)
            self.val_features = scaler.transform(val_features)

            train_dataset = CustomDataset(self.train_features, train_target)
            val_dataset = CustomDataset(self.val_features, val_target)

            batch_size = self.model_config["batch_size"]
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)


          def load_model_object(self):
            onnx_model = onnx.load(self.model_config["saved_model_path"])
            model = ConvertModel(onnx_model,experimental=True)
            self.model=model


          def load_model_optimizer(self):
            #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.optimizer = optim.Adam(self.model.parameters())
            #optimizer=torch.load(self.model_config["saved_model_optimizer"])
            #self.optimizer=optimizer

          def make_dprivate(self):
            privacy_engine = PrivacyEngine()
            modules=privacy_engine.make_private_with_epsilon(module=self.model,optimizer=self.optimizer,data_loader=self.train_loader,target_epsilon=self.model_config["epsilon_threshold"],target_delta=self.model_config["delta"],epochs=self.model_config["total_epochs"],max_grad_norm=self.model_config["max_grad_norm"],batch_first="True")
            self.model=modules[0]
            self.optimizer=modules[1]
            self.train_loader=modules[2]

          def execute_model(self):
            self.logger_list=[]
            criterion = nn.MSELoss()
            for epoch in range(self.model_config["total_epochs"]):
              for inputs, labels in self.train_loader:
                  self.optimizer.zero_grad()
                  outputs = self.model(inputs)
                  loss = criterion(outputs, labels.unsqueeze(1))
                  loss.backward()
                  self.optimizer.step()
              self.logger_list.append('Epoch [{epoch+1}/{self.model_config["total_epochs"]}], Loss: {loss.item():.4f}')
              print(f'Epoch [{epoch+1}/{self.model_config["total_epochs"]}], Loss: {loss.item():.4f}')
            output_path=self.model_config["trained_model_output_path"]
            print('Writing training model to ' + output_path)
            torch.onnx.export(self.model, torch.randn(1, self.train_features.shape[1]), output_path, verbose=True)

          def ccr_model_run(self):
            self.load_data()
            self.load_model_object()
            self.load_model_optimizer()
            self.make_dprivate()
            self.execute_model()


def ccr_logger_function(ccr_tracking_object,ccr_model):
  """
    Function to implement logging for audit/model cert
  """
  file_path=ccr_tracking_object["ccr_pbt_logger_file"]
  with open(file_path, 'w') as file:
    file.write("Model Architecture\n")
    string=str(ccr_model.model)
    file.write(string)
    for c in ccr_model.logger_list:
      file.write(c)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=Path, help='Model configuration')
    args = parser.parse_args()
    model_config = json.loads(args.model_config.read_text())
    print(model_config)
    args = parser.parse_args(argv)

    ccr_private_model=ccr_model(model_config)
    ccr_private_model.ccr_model_run()
    ccr_logger_function(logger,ccr_private_model)

if __name__ == '__main__':
    main()
