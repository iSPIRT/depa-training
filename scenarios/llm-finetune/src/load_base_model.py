from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch
# import os
from pathlib import Path
from ast import literal_eval
from huggingface_hub import login

from torch.quantization import quantize_dynamic
import json

# Load config from JSON
CONFIG_PATH = "/mnt/config/model_repo_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Class for handling language model loading and saving
class ModelHandler:
    def __init__(self, model_name, hf_token, q_prec, save_dir="saved_models", device = 'cuda'):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        login(hf_token)
        self.q_prec = q_prec
        if not torch.cuda.is_available() and device == 'cuda':
            print("CUDA is not available. Using CPU instead.")
            self.device = 'cpu'
        else:
            self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        
        
    def load_from_huggingface(self):
        """Load model and tokenizer from HuggingFace using torch.quantization for CPU"""

        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load full-precision model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config
        )

        # Apply dynamic quantization for CPU
        self.model = quantize_dynamic(
            self.model,  # the original model
            {torch.nn.Linear},  # only quantize Linear layers
            dtype=torch.qint8  # use 8-bit integers
        )

        # Move to CPU explicitly
        self.model.to(torch.device(self.device))

        # Ensure padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.model, self.tokenizer

    
    def save_locally(self, format="pytorch"):
        """Save model and tokenizer locally in specified format"""
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise ValueError("Model and tokenizer must be loaded first")
            
        # Create model-specific directory
        model_save_dir = self.save_dir / self.model_name.replace('/', '_')
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "pytorch":
            # Save in PyTorch format
            torch.save(self.model.state_dict(), 
                      model_save_dir / "model.pth")
            self.tokenizer.save_pretrained(model_save_dir)
            self.config.save_pretrained(model_save_dir)
            
        # elif format.lower() == "onnx":
        #     # Save in ONNX format
        #     dummy_input = torch.zeros(1, 512, dtype=torch.long)
        #     torch.onnx.export(self.model, 
        #                     dummy_input,
        #                     model_save_dir / "model.onnx",
        #                     export_params=True,
        #                     opset_version=11)
        #     self.tokenizer.save_pretrained(model_save_dir)
        #     self.config.save_pretrained(model_save_dir)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Model and tokenizer saved in {format} format at {model_save_dir}")
        return str(model_save_dir)

    def download_from_huggingface(self):
        # Create model-specific directory
        model_save_dir = self.save_dir / self.model_name.replace('/', '_')
        model_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {self.model_name} from HuggingFace...")

        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load full-precision model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config
        )
        
        # Save original model (not quantized)
        self.model.save_pretrained(model_save_dir)
        self.tokenizer.save_pretrained(model_save_dir)
        self.config.save_pretrained(model_save_dir)

        print(f"Model and tokenizer saved to {model_save_dir}")
    

    def load_locally(self, local_dir=None, format="pytorch"):
        """Load model and tokenizer from local directory"""
        if local_dir is None:
            local_dir = self.save_dir / self.model_name.replace('/', '_')
        else:
            local_dir = Path(local_dir)
            
        if not local_dir.exists():
            raise ValueError(f"Directory not found: {local_dir}")
            
        # Load configuration and tokenizer
        self.config = AutoConfig.from_pretrained(local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        
        # Create model instance
        self.model = AutoModelForCausalLM.from_config(self.config)
        
        if format.lower() == "pytorch":
            # Load PyTorch weights
            state_dict = torch.load(local_dir / "model.pth")
            self.model.load_state_dict(state_dict)
            
        # elif format.lower() == "onnx":
        #     # For ONNX, we need to use ONNX Runtime
        #     import onnxruntime as ort
        #     self.model = ort.InferenceSession(str(local_dir / "model.onnx"))
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Ensure padding token is set as EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer


# MODEL_NAME = "meta-llama/Llama-3.2-1b"
MODEL_NAME = config.get("MODEL_NAME", "facebook/opt-350m")
HF_READ_TOKEN = config.get("REPO_READ_TOKEN", "")
SAVE_DIR = config.get("SAVE_DIR", "/mnt/model")
Q_PREC = config.get("Q_PREC", "load_in_4bit")

handler = ModelHandler(model_name=MODEL_NAME, save_dir=SAVE_DIR, hf_token=HF_READ_TOKEN, q_prec=Q_PREC, device='cuda')

# Download and save the model
handler.download_from_huggingface()

# # Load from HuggingFace
# model, tokenizer = handler.load_from_huggingface()

# # Save locally in PyTorch format (NOTE: NEED TO CHANGE TO ONNX!)
# save_path = handler.save_locally(format="pytorch")

# # Load from local storage
# model, tokenizer = handler.load_locally(format="pytorch")