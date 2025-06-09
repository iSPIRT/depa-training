import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
from pathlib import Path
# import os
# from typing import Optional, Dict, Union#, List
# import pickle
import json

# Load config from JSON
CONFIG_PATH = "mnt/config/medquad_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Created specifically LLM-Finetune scenario

class DatasetHandler:
    def __init__(self, dataset_name: str, dataset_split: str = None):
        """
        Initialize the DatasetHandler.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            save_dir: Directory to save datasets locally
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        
    def load_from_huggingface(self) -> Dataset:
        """Load dataset from HuggingFace Hub"""
        dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        self.dataset = dataset['train'] if isinstance(dataset, dict) else dataset
        return self.dataset
    
    def save_locally(self, save_dir: str, format: str = "csv") -> str:
        """
        Save dataset locally in specified format.
        
        Args:
            format: Format to save the dataset in ('parquet', 'csv', 'json', 'pickle')
            
        Returns:
            Path to saved dataset
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset must be loaded first")
            
        save_dir = Path(save_dir)
        # Create dataset-specific directory
        # dataset_save_dir = save_dir / self.dataset_name.split('/')[0]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # # Save dataset info
        # dataset_info = {
        #     'name': self.dataset_name,
        #     'format': format,
        #     'splits': list(self.dataset.keys()) if isinstance(self.dataset, dict) else ['train']
        # }
        # with open(dataset_save_dir / 'dataset_info.json', 'w') as f:
        #     json.dump(dataset_info, f)
        
        # splits_to_save = [split] if split else (dataset_info['splits'])
        
        # for split_name in splits_to_save:
        #     split_data = self.dataset[split_name] if isinstance(self.dataset, dict) else self.dataset
        #     save_path = dataset_save_dir / f"{split_name}.{format}"

        save_path = save_dir / f"{self.dataset_name.split('/')[1]}.csv"
            
        if format == "csv":
            # Save as CSV
            df = pd.DataFrame(self.dataset)
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='any')
            for col in ("Question", "Answer"):
                df[col] = df[col].astype(str)
            df.to_csv(save_path, index=False)
                
        # elif format == "parquet":
        #     # Save as Parquet (recommended for efficiency)
        #     table = pa.Table.from_batches([pa.record_batch(split_data)])
        #     pq.write_table(table, save_path)
            
        # elif format == "json":
        #     # Save as JSON
        #     with open(save_path, 'w') as f:
        #         json.dump(split_data.to_dict(), f)
                
        # elif format == "pickle":
        #     # Save as Pickle
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(split_data, f)
                    
        else:
            raise ValueError(f"Unsupported format: {format}")
                
        print(f"Dataset saved in {format} format at {save_dir}")
        return str(save_dir)
    
    def load_locally(self, local_dir, format) -> Dataset:
        """
        Load dataset from local storage.
        
        Args:
            local_dir: Directory containing the saved dataset
            format: Format of the saved dataset.
            
        Returns:
            Loaded dataset
        """
        # if local_dir is None:
        #     local_dir = self.save_dir / self.dataset_name.split('/')[0]
        # else:
        #     local_dir = Path(local_dir)
            
        if not local_dir.exists():
            raise ValueError(f"Directory not found: {local_dir}")
            
        # # Load dataset info
        # with open(local_dir / 'dataset_info.json', 'r') as f:
        #     dataset_info = json.load(f)
            
        # if format is None:
        #     format = dataset_info['format']
            
        # splits_to_load = [split] if split else dataset_info['splits']
        # loaded_datasets = {}
        
        # for split_name in splits_to_load:
        #     file_path = local_dir / f"{split_name}.{format}"

        file_path = local_dir / f"{self.dataset_name.split('/')[1]}.csv"
            
        if format == "csv":
            # Load from CSV
            df = pd.read_csv(file_path)
            self.dataset = Dataset.from_pandas(df)
            
        # elif format == "parquet":
        #     # Load from Parquet
        #     table = pq.read_table(file_path)
        #     loaded_data = Dataset.from_pandas(table.to_pandas())
            
        # elif format == "json":
        #     # Load from JSON
        #     with open(file_path, 'r') as f:
        #         data_dict = json.load(f)
        #     loaded_data = Dataset.from_dict(data_dict)
            
        # elif format == "pickle":
        #     # Load from Pickle
        #     with open(file_path, 'rb') as f:
        #         loaded_data = pickle.load(f)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # loaded_datasets[split_name] = loaded_data
            
        # self.dataset = loaded_datasets if len(splits_to_load) > 1 else loaded_datasets[splits_to_load[0]]

        return self.dataset

# Example usage:
DATASET_NAME = config.get("DATASET_NAME", "keivalya/MedQuad-MedicalQnADataset")
SAVE_DIR = config.get("SAVE_DIR", '/mnt/output/medquad/')
FORMAT = config.get("FORMAT", "csv")

handler = DatasetHandler(dataset_name=DATASET_NAME)

# Load from HuggingFace
dataset = handler.load_from_huggingface()

# Save locally in Parquet format (recommended)
save_path = handler.save_locally(save_dir=SAVE_DIR, format=FORMAT)

# Load from local storage
# dataset = handler.load_locally(local_dir=SAVE_DIR, format=FORMAT)