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


### NEW IMPORTS ###

# For parsing configs and args
import argparse
import json
import os
# For standard ML procedures
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.quantization import quantize_dynamic  # For dynamic quantization
# For loading readymade models and datasets
from datasets import load_dataset
from datasets import Dataset as ds
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_scheduler
# For parameter-efficient training
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
# For differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes
# For monitoring training progress
from tqdm import tqdm
# For accessing HuggingFace Hub
from huggingface_hub import login#, HfApi, HfFolder, Repository

###################


### OLD IMPORTS ###

# # torch related imports
# from typing import Optional
# import torch
# from torchvision import datasets, transforms

# from tqdm import tqdm
# import torch.utils.data as data
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# # sklearn,pandas,numpy related imports
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd

# # opacus related imports
# from opacus.accountants import create_accountant
# from opacus import PrivacyEngine

# # onnx related imports
# import onnx
# from onnx2pytorch import ConvertModel

# other imports
# import os
# import json
# import argparse
# from pathlib import Path

###################

from .task_base import TaskBase

logger = {
    "epochs_per_report": 1,
    "metrics": [
        "tdp_config",
        "tdc_config",
        "model_architecture",
        "model_hyperparameters",
        "model_config",
        "accuracy",
        "precision",
        "recall",
    ],
    "ccr_pbt_logger_file": "/mnt/remote/output/ccr_depa_trg_model_logger.json",
}

# def compute_delta(ccr_context):
#     return 1 / ccr_context["sample_size"]


# class CustomDataset(Dataset):
#     """
#     Class to convert dataset columns to tensors
#     """

#     def __init__(self, features, target):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.target = torch.tensor(target.values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.target[idx]


class PrivateLLMFineTune(TaskBase):
    """
    Args:
    config:training configuration 

    Methods:
    load_data:loads data from HuggingFace repo, tokenizes and prepares dataloaders for training
    load_model:loads model object from model config
    load_optimizer:loads model optimizer from model config
    apply_lora:applies lora to the model using peft
    make_dprivate:make model,dataloader and optimizer private
    train:differentially private llm finetuning
    save_model_ft:saves and pushes model to HuggingFace repo
    execute:mega function which includes all the above functions

    """

    def init(self, config):
        # self.DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
        self.device = torch.device(config["device"])

        self.config = config
        self.model = None
        self.model_config = None
        self.tokenizer = None
        self.dataset = None
        self.train_loader = None
        # self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None
        self.loss_fn = CrossEntropyLoss()
        self.model_ft = None

        # login(config["HF_READ_TOKEN"])

        print("*** STARTING FINE-TUNING ***")
        print(f"Fine-tuning {config['model_name']} on joined datasets")#{config['DATASET_NAME']}...")

    # def ccr_logger_function(ccr_tracking_object, ccr_model):
    #     """
    #     Function to implement logging for audit/model cert
    #     """
    #     file_path = ccr_tracking_object["ccr_pbt_logger_file"]
    #     with open(file_path, "w") as file:
    #         file.write("Model Architecture\n")
    #         string = str(ccr_model.model)
    #         file.write(string)
    #         for c in ccr_model.logger_list:
    #             file.write(c)


    def load_model(self):
        # Load pretrained model configurations and tokenizer
        # local_dir = self.config["saved_model_dir"] + self.config["model_name"].replace("/", "_")
        local_dir = os.path.join(self.config["saved_model_dir"], self.config["model_name"].replace("/", "_"))

        print(f"Debug | load_model | local model dir: {local_dir}")

        # Load configuration and tokenizer
        if not os.path.exists(local_dir):
            raise ValueError(f"Directory not found: {local_dir}")

        # Load configuration and tokenizer
        # self.model_config = AutoConfig.from_pretrained(local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(local_dir)
        
        print("Debug | load_model | base model instantiation")

        # Apply quantization if requested
        if self.config["Q_PRECISION"] != "none":
            if self.config["Q_PRECISION"] == "8bit":
                self.model = quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                # prepare the model for k-bit training
                self.model = prepare_model_for_kbit_training(self.model)
            elif self.config["Q_PRECISION"] == "4bit":
                self.model = quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint4
                )
                self.model = prepare_model_for_kbit_training(self.model)
        
        # Move to target device
        self.model.to(torch.device(self.device))

        # state_dict = torch.load(os.path.join(local_dir, "model.pth"))
        # self.model.load_state_dict(state_dict)

        print("Debug | load_model | base model successfully loaded")

        # Ensure padding token is set as EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def load_data(self):
        # Load the CSV file
        df = pd.read_csv(self.config["input_dataset_path"])
        df = df.head(40)  # for testing, select only first 400 rows
        print("Debug | load_data | Loaded joined dataset")

        # Create dataset class
        class QADataset(Dataset):
            def __init__(self, outer_parent, inputs, outputs, max_length):
                self.inputs = inputs
                self.outputs = outputs
                self.tokenizer = outer_parent.tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                # Tokenize input and output
                input_encoding = self.tokenizer(
                    self.inputs[idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                output_encoding = self.tokenizer(
                    self.outputs[idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": input_encoding["input_ids"].squeeze(),
                    "attention_mask": input_encoding["attention_mask"].squeeze(),
                    "labels": output_encoding["input_ids"].squeeze()
                }
        
        # Create dataset and dataloader
        self.dataset = QADataset(self, df["input"].tolist(), df["output"].tolist(), 512)
        print("Debug | load_data | Tokenized inputs and outputs")
        self.train_loader = DataLoader(self.dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True, drop_last=True)
        print("Debug | load_data | Trainloader prepared")
    

    def load_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["LEARNING_RATE"])
        self.scheduler = get_scheduler(
            name="linear",  # You can also use "cosine" or other schedules
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config["NUM_EPOCHS"] * len(self.train_loader)  # Total number of training steps,
        )
        print("Debug | load_optimizer_and_scheduler | Optimizer and Scheduler loaded")


    def apply_lora(self):
        lora_config = LoraConfig(
            r=self.config["LORA_RANK"], # 8, # Rank of the low-rank matrices
            lora_alpha=self.config["LORA_ALPHA"], # 32, # Scaling factor for the LoRA updates
            target_modules=self.config["LORA_TARGET_MODULES"], # ["q_proj", "v_proj"], # Modules to apply LoRA to  ### Modify as per model architecture
            lora_dropout=self.config["LORA_DROPOUT"], # 0.05, # Dropout probability applied to the LoRA updates for regularization
            bias=self.config["LORA_BIAS"], # "none", # Whether to include bias parameters in the LoRA layers
            task_type="CAUSAL_LM" # Type of task - eg. causal modelling, seq2seq
        )
    
        # Obtain the parameter-efficient LoRA model
        self.model = get_peft_model(self.model, lora_config)
        print("Debug | apply_lora | LoRA hooks applied")


    def make_dprivate(self):
        self.privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed
        self.model.train()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config["NUM_EPOCHS"],
            target_delta=self.config["DELTA"],  # Privacy budget
            target_epsilon=self.config["EPSILON"],  # Probability of privacy breach
            max_grad_norm=self.config["MAX_GRAD_NORM"], # threshold for clipping the norm of per-sample gradients
            poisson_sampling=False
        )

        print("Debug | make_dprivate | Opacus PrivacyEngine hooks applied")

    def train(self):
        print("Debug | train | Begin fine-tuning")
        # 8. Training loop with BatchMemoryManager
        self.model.train()
        for epoch in range(1, self.config["NUM_EPOCHS"] + 1):
            losses = []

            # Use BatchMemoryManager for managing memory
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=self.config["MAX_PHYSICAL_BATCH_SIZE"],
                optimizer=self.optimizer
            ) as memory_safe_loader:

                # Training step
                for step, batch in enumerate(tqdm(memory_safe_loader, desc=f"Epoch {epoch}/{self.config['NUM_EPOCHS']}")):
                    self.optimizer.zero_grad()
                    # Move batch to DEVICE

                    # input_ids, attention_mask, labels = batch
                    # input_ids = input_ids.to(self.device)
                    # attention_mask = attention_mask.to(self.device)
                    # labels = labels.to(self.device)

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # If empty batches are there, skip them
                    if input_ids.size(0) == 0:
                        continue

                    # Fwd pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits  # Model predictions
            
                    # compute loss
                    # shift logits and labels for causal language modeling
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
                    # Bkwd pass and model optim
                    # self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
                    losses.append(loss.item())

                    # log progress every 50 steps
                    if step > 0 and step % 50 == 0:
                        train_loss = np.mean(losses)
                        # epsilon = self.privacy_engine.get_epsilon(self.config["DELTA"])

                        print(
                            f"Epoch: {epoch} | Step: {step} | "
                            f"Train loss: {train_loss:.3f} | "
                            # f"ɛ: {epsilon:.2f}"
                        )

            # epoch summary
            train_loss = np.mean(losses)
            # epsilon = self.privacy_engine.get_epsilon(self.config["DELTA"])
            print(f"Epoch {epoch} completed. Average loss: {train_loss:.4f}")#, ɛ: {epsilon:.2f}")

        print("Debug | train | Fine-tuning completed")


        # 9. Unwrap the DP fine-tuned model - the model is currently wrapped by a PEFT wrapper as well as an Opacus wrapper

        ## Step 1: Check if the model is wrapped in GradSampleModule (Opacus wrapper)
        if isinstance(self.model, opacus.grad_sample.GradSampleModule):
            unwrapped_model = self.model._module  # Access the underlying model from GradSampleModule
        else:
            unwrapped_model = self.model  # If not wrapped, use the model as-is
        ## Step 2: For LoRA/PEFT models, unwrap further
        if isinstance(unwrapped_model, peft.PeftModelForCausalLM):
            self.model_ft = unwrapped_model.base_model  # Extract the base model under the PEFT wrapper
        else:
            self.model_ft = unwrapped_model  # If not a PEFT model, use as-is
        
        # Set model for inference by freezing parameters
        self.model_ft.eval()
        


    def save_ft_model_hf(self):
        # 10. Push to HuggingFace Hub
        self.model_ft.push_to_hub(self.config["SAVE_MODEL_REPO_NAME"], token=self.config["HF_WRITE_TOKEN"])
        self.tokenizer.push_to_hub(self.config["SAVE_MODEL_REPO_NAME"], token=self.config["HF_WRITE_TOKEN"])
        print(f"Model and tokenizer pushed to Hugging Face Hub: https://huggingface.co/{self.config['SAVE_MODEL_REPO_NAME']}")


    def save_ft_model_local(self):
        # 10. Save the fine-tuned model locally
        local_dir = os.path.join(self.config["trained_model_output_path"])
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        self.model_ft.save_pretrained(local_dir)
        self.tokenizer.save_pretrained(local_dir)
        print(f"Fine-tuned model and tokenizer saved locally at: {local_dir}")
        
        # Optionally, save the model in ONNX format
        # onnx_path = os.path.join(local_dir, "model.onnx")
        # torch.onnx.export(self.model_ft, dummy_input, onnx_path)
        # print(f"Model saved in ONNX format at: {onnx_path}")


    def execute(self, config):
        try:
            # --- START OF FINE-TUNING CODE ---
            self.init(config)
            self.load_model()
            self.load_data()
            self.load_optimizer_and_scheduler()
            self.apply_lora()
            self.make_dprivate()
            self.train()
            print("Fine-tuning complete!")
            # --- END OF FINE-TUNING CODE ---

            # NOTE: Approach is to work on the local for saving the model
            # If required, can be pushed to any model repo like HuggingFace
            # self.save_ft_model_hf()
            self.save_ft_model_local()

        except Exception as e:
            print(f"An error occurred during fine-tuning: {e}")
            exit(1)