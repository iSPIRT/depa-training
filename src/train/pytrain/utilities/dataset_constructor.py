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
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OrdinalEncoder
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from safetensors.torch import load_file as st_load
import h5py

def encode_categoricals(df: pd.DataFrame, cat_cols: list):
    """
    Encode categorical columns:
      - If cardinality <= low_card_threshold -> one-hot (get_dummies)
      - Else -> OrdinalEncoder with unknown_value = -1
    Returns transformed dataframe and list of new feature column names.
    """
    low_card_threshold = max(2, int(0.01 * len(df)))
    df = df.copy()
    encoded_parts = []
    kept_cols = []

    high_card_cols = []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=False)
        if nunique <= low_card_threshold:
            # one-hot
            d = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=True)
            encoded_parts.append(d)
            kept_cols += d.columns.tolist()
        else:
            high_card_cols.append(c)

    if high_card_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        # sklearn expects 2D array
        arr = oe.fit_transform(df[high_card_cols].astype(str))
        df_enc = pd.DataFrame(arr, columns=[f"{c}_ord" for c in high_card_cols], index=df.index)
        encoded_parts.append(df_enc)
        kept_cols += df_enc.columns.tolist()

    if encoded_parts:
        encoded_df = pd.concat(encoded_parts, axis=1)
    else:
        encoded_df = pd.DataFrame(index=df.index)

    return encoded_df, kept_cols


def build_feature_matrix(df: pd.DataFrame, num_cols: list, cat_encoded_df: pd.DataFrame):
    """Stack numeric cols + encoded categorical df into final X matrix"""
    parts = []
    if num_cols:
        parts.append(df[num_cols].reset_index(drop=True))
    if not cat_encoded_df.empty:
        parts.append(cat_encoded_df.reset_index(drop=True))
    if parts:
        X = pd.concat(parts, axis=1)
    else:
        raise ValueError("No features available after preprocessing.")
    return X

def create_dataset(config: str, data_path: str) -> Dict[str, Dataset]:
    """
    Create all dataset splits based on configuration file
    
    Args:
        config: JSON configuration dictionary
        
    Returns:
        Dict containing all available splits {'train': dataset, 'val': dataset, 'test': dataset}
    """
    
    dataset_type = config.get('type', 'tabular')
    
    if dataset_type == 'tabular':
        dataset = TabularDataset(config, data_path)
    elif dataset_type == 'directory':
        dataset = DirectoryDataset(config, data_path)
    elif dataset_type == 'serialized':
        dataset = SerializedDataset(config, data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset.get_all_splits()

class BaseDataset:
    """Base class with common functionality for all dataset types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transform = None
        self.scaler = None
        self.splits_data = {}
        
        # Initialize transforms if specified
        self._setup_transforms()
        self._setup_preprocessing()
        
    def _setup_transforms(self):
        """Setup data transforms based on config"""
        transform_config = self.config.get('transforms', {})
        # This is a placeholder - you can extend with actual transform implementations
        pass
    
    def _setup_preprocessing(self):
        """Setup preprocessing/normalization based on config"""
        preprocessing_config = self.config.get('preprocessing', {})
        scaler_type = preprocessing_config.get('scaler', None)
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'normalizer':
            self.scaler = Normalizer()
    
    def _create_splits(self, data, targets=None):
        """Create train/val/test splits based on config"""
        split_config = self.config.get('splits', {})
        
        if not split_config:
            # If no splits specified, return all data as train
            return {'train': (data, targets) if targets is not None else data}
        
        train_ratio = split_config.get('train', 1.0)
        val_ratio = split_config.get('val', 0.0) 
        test_ratio = split_config.get('test', 0.0)
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        if targets is not None:
            # For supervised learning
            splits = {}
            remaining_data = data
            remaining_targets = targets
            
            if test_ratio > 0:
                # Create test split first if specified
                remaining_data, X_test, remaining_targets, y_test = train_test_split(
                    data, targets, test_size=test_ratio,
                    random_state=split_config.get('random_state', 42),
                    stratify=targets if split_config.get('stratify', False) else None
                )
                splits['test'] = (X_test, y_test)
            
            if val_ratio > 0:
                # Create validation split if specified
                X_train, X_val, y_train, y_val = train_test_split(
                    remaining_data, remaining_targets, 
                    test_size=val_ratio/(val_ratio + train_ratio),
                    random_state=split_config.get('random_state', 42),
                    stratify=remaining_targets if split_config.get('stratify', False) else None
                )
                splits['train'] = (X_train, y_train)
                splits['val'] = (X_val, y_val)
            else:
                # Just use remaining data as train
                splits['train'] = (remaining_data, remaining_targets)

            final_splits = {'train': splits['train']}
            if 'val' in splits:
                final_splits['val'] = splits['val']
            if 'test' in splits:
                final_splits['test'] = splits['test']
                
            return final_splits
            
        else:
            # For unsupervised learning or when targets are not separate
            splits = {}
            indices = list(range(len(data)))
            remaining_indices = indices
            
            if test_ratio > 0:
                # Create test split first if specified
                remaining_indices, test_indices = train_test_split(
                    indices, test_size=test_ratio,
                    random_state=split_config.get('random_state', 42)
                )
                splits['test'] = [data[i] for i in test_indices]
            
            if val_ratio > 0:
                # Create validation split if specified
                train_indices, val_indices = train_test_split(
                    remaining_indices, 
                    test_size=val_ratio/(val_ratio + train_ratio),
                    random_state=split_config.get('random_state', 42)
                )
                splits['train'] = [data[i] for i in train_indices]
                splits['val'] = [data[i] for i in val_indices]
            else:
                # Just use remaining indices as train
                splits['train'] = [data[i] for i in remaining_indices]
                
            return splits
    
    def get_all_splits(self):
        """Return all available dataset splits"""
        return self.splits_data

class SplitDataset(Dataset):
    """Individual dataset for a specific split"""
    
    def __init__(self, features, targets=None, transform=None, scaler=None, fit_scaler=False, data_type='numpy', encoding_info: Dict = None):
        self.transform = transform
        self.scaler = scaler
        self.data_type = data_type
        self.encoding_info = encoding_info or {}
        
        # Apply preprocessing to features
        if self.scaler:
            if fit_scaler:
                self.features = self.scaler.fit_transform(features)
            else:
                self.features = self.scaler.transform(features)
        else:
            self.features = features
        
        if self.data_type == 'tensor':
            # Convert to tensors if numpy arrays
            if isinstance(self.features, np.ndarray):
                self.features = torch.tensor(self.features, dtype=torch.float32)
            else:
                self.features = self.features

            if targets is not None:
                if isinstance(targets, np.ndarray):
                    self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)  # Reshape to match model output
                else:
                    self.targets = targets
            else:
                self.targets = None

        if self.data_type == 'numpy':
            if not isinstance(self.features, np.ndarray):
                self.features = np.array(self.features)
            else:
                self.features = self.features
                
            if targets is not None:
                if not isinstance(targets, np.ndarray):
                    self.targets = np.array(targets)
                else:
                    self.targets = targets
            else:
                self.targets = None
    
    def get_encoding_info(self) -> Dict:
        """Return information about how categorical features were encoded"""
        return self.encoding_info.copy()
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after encoding"""
        if self.encoding_info:
            return (self.encoding_info.get('numerical_columns', []) + 
                   self.encoding_info.get('encoded_columns', []))
        return []
    
    def get_categorical_columns(self) -> List[str]:
        """Return list of original categorical column names"""
        return self.encoding_info.get('categorical_columns', [])
    
    def get_numerical_columns(self) -> List[str]:
        """Return list of numerical column names"""
        return self.encoding_info.get('numerical_columns', [])
    
    def get_encoded_columns(self) -> List[str]:
        """Return list of encoded categorical column names"""
        return self.encoding_info.get('encoded_columns', [])
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return dictionary with feature dimensions for different types"""
        return {
            'total_features': len(self.get_feature_names()),
            'numerical_features': len(self.get_numerical_columns()),
            'categorical_features': len(self.get_categorical_columns()),
            'encoded_features': len(self.get_encoded_columns())
        }
    
    def __len__(self):
        if isinstance(self.features, (list, tuple)):
            return len(self.features)
        elif hasattr(self.features, '__len__'):
            return len(self.features)
        else:
            return self.features.size(0)
    
    def __getitem__(self, idx):
        if isinstance(self.features, (list, tuple)):
            features = self.features[idx]
        else:
            features = self.features[idx]
            
        if self.transform:
            features = self.transform(features)
            
        if self.targets is not None:
            if isinstance(self.targets, (list, tuple)):
                targets = self.targets[idx]
            else:
                targets = self.targets[idx]
            return features, targets
        else:
            return features

class TabularDataset(BaseDataset):
    """Dataset for tabular data (CSV, Excel, etc.)"""
    
    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config)
        self.data_path = data_path
        self.target_variable = config.get('target_variable')
        self.feature_columns = config.get('feature_columns', None)
        self.data_type = config.get('data_type', 'numpy')
        self.categorical_columns = []
        self.numerical_columns = []
        self.encoding_info = {}
        self._load_and_split_data()
        
    def _load_and_split_data(self):
        """Load and preprocess tabular data, then create splits"""
        file_ext = Path(self.data_path).suffix.lower()
        
        # Load data based on file extension
        if file_ext == '.csv':
            data = pd.read_csv(self.data_path)
        elif file_ext in ['.xlsx', '.xls']:
            data = pd.read_excel(self.data_path)
        elif file_ext == '.parquet':
            data = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Handle missing values
        missing_strategy = self.config.get('missing_strategy', 'drop')
        if missing_strategy == 'drop':
            data = data.dropna()
        elif missing_strategy == 'fill':
            fill_value = self.config.get('fill_value', 0)
            data = data.fillna(fill_value)
        elif missing_strategy == 'forward_fill':
            data = data.fillna(method='ffill')
        elif missing_strategy == 'backward_fill':
            data = data.fillna(method='bfill')
        
        # Select features
        if self.feature_columns:
            features = data[self.feature_columns]
        elif self.target_variable:
            features = data.drop(columns=[self.target_variable])
        else:
            features = data
        
        # Extract target
        targets = data[self.target_variable].values if self.target_variable else None
        
        # Identify categorical and numerical columns
        self.categorical_columns = features.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Encode categorical features
        if self.categorical_columns:
            encoded_df, kept_cols = encode_categoricals(features, self.categorical_columns)
            self.encoding_info['encoded_columns'] = kept_cols
            self.encoding_info['categorical_columns'] = self.categorical_columns
            self.encoding_info['numerical_columns'] = self.numerical_columns
        else:
            encoded_df = pd.DataFrame(index=features.index)
            self.encoding_info['encoded_columns'] = []
            self.encoding_info['categorical_columns'] = []
            self.encoding_info['numerical_columns'] = self.numerical_columns
        
        # Build the final feature matrix
        X = build_feature_matrix(features, self.numerical_columns, encoded_df)
        
        # Create splits
        splits = self._create_splits(X, targets)
        
        # Create dataset objects for each split
        for split_name, (split_features, split_targets) in splits.items():
            fit_scaler = (split_name == 'train')  # Only fit scaler on training data
            self.splits_data[split_name] = SplitDataset(
                features=split_features, 
                targets=split_targets, 
                transform=self.transform, 
                scaler=self.scaler,
                fit_scaler=fit_scaler,
                data_type=self.data_type,
                encoding_info=self.encoding_info
            )

class DirectoryDataset(BaseDataset):
    """Dataset for directory-based data (images, documents, etc.)"""
    
    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config)
        self.data_dir = data_path
        self.file_pattern = config.get('file_pattern', '*')
        self.data_type = config.get('data_type', 'image')
        
        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """Load directory-based data and create splits"""
        structure_type = self.config.get('structure_type', 'flat')
        
        if structure_type == 'flat':
            data = self._load_flat_structure()
        elif structure_type == 'nested':
            data = self._load_nested_structure()
        elif structure_type == 'paired':
            data = self._load_paired_structure()
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
        
        # Create splits
        splits = self._create_splits(data)
        
        # Create dataset objects for each split
        for split_name, split_data in splits.items():
            self.splits_data[split_name] = DirectorySplitDataset(
                split_data, 
                data_type=self.data_type,
                config=self.config,
                transform=self.transform
            )
    
    def _load_flat_structure(self):
        """Load files from flat directory structure"""
        return glob(os.path.join(self.data_dir, self.file_pattern))
    
    def _load_nested_structure(self):
        """Load files from nested directory structure (e.g., class folders)"""
        class_folders = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d))]
        
        samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_folders))}
        
        for class_name in class_folders:
            class_dir = os.path.join(self.data_dir, class_name)
            file_paths = glob(os.path.join(class_dir, self.file_pattern))
            
            for file_path in file_paths:
                samples.append((file_path, self.class_to_idx[class_name]))
        
        return samples
    
    def _load_paired_structure(self):
        """Load paired files (e.g., image-mask pairs like BRATS)"""
        pairing_config = self.config.get('pairing', {})
        input_pattern = pairing_config.get('input_pattern', '*')
        target_pattern = pairing_config.get('target_pattern', '*')
        
        samples = []
        
        # Handle BRATS-like structure
        if 'folder_pattern' in pairing_config:
            folder_pattern = pairing_config['folder_pattern']
            patient_folders = glob(os.path.join(self.data_dir, folder_pattern))
            
            for patient_folder in patient_folders:
                patient_id = os.path.basename(patient_folder)
                input_files = sorted(glob(os.path.join(patient_folder, 
                                                     input_pattern.replace('*', patient_id))))
                
                for input_file in input_files:
                    # Extract identifier for pairing
                    base_name = os.path.basename(input_file)
                    identifier = self._extract_identifier(base_name, patient_id, pairing_config)
                    
                    # Find corresponding target file
                    target_file = os.path.join(patient_folder, 
                                             target_pattern.replace('*', patient_id).replace('{id}', identifier))
                    
                    if os.path.exists(target_file):
                        # Optional filtering (e.g., non-empty masks)
                        if self._should_include_sample(input_file, target_file):
                            samples.append((input_file, target_file))
        
        return samples
    
    def _extract_identifier(self, filename: str, patient_id: str, pairing_config: Dict) -> str:
        """Extract identifier for file pairing"""
        # Remove patient ID and extract slice/identifier info
        identifier_pattern = pairing_config.get('identifier_extraction', {})
        
        if 'remove_prefix' in identifier_pattern:
            filename = filename.replace(identifier_pattern['remove_prefix'], '')
        if 'remove_suffix' in identifier_pattern:
            filename = filename.replace(identifier_pattern['remove_suffix'], '')
        
        return filename
    
    def _should_include_sample(self, input_file: str, target_file: str) -> bool:
        """Determine if sample should be included (e.g., non-empty masks)"""
        filter_config = self.config.get('filtering', {})
        
        if filter_config.get('filter_empty_targets', False):
            if self.data_type == 'image':
                target_img = cv2.imread(target_file)
                return not np.all(target_img == 0)
        
        return True

class DirectorySplitDataset(Dataset):
    """Individual dataset for directory-based data splits"""
    
    def __init__(self, data, data_type='image', config=None, transform=None):
        self.data = data
        self.data_type = data_type
        self.config = config or {}
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if isinstance(sample, tuple) and len(sample) == 2:
            # Check if it's (file_path, label) or (input_path, target_path)
            first_item, second_item = sample
            if isinstance(second_item, (int, float)):
                # It's (file_path, label)
                data = self._load_file(first_item)
                return data, second_item
            else:
                # It's (input_path, target_path)
                input_data = self._load_file(first_item)
                target_data = self._load_file(second_item)
                return input_data, target_data
        else:
            # Single file path
            return self._load_file(sample)
    
    def _load_file(self, file_path: str):
        """Load individual file based on data type"""
        if self.data_type == 'image':
            return self._load_image(file_path)
        elif self.data_type == 'text':
            return self._load_text(file_path)
        elif self.data_type == 'audio':
            return self._load_audio(file_path)
        else:
            # Generic file loading
            with open(file_path, 'rb') as f:
                return f.read()
    
    def _load_image(self, image_path: str):
        """Load and preprocess image"""
        image_config = self.config.get('image_config', {})
        
        if image_config.get('use_cv2', False):
            img = cv2.imread(image_path)
            if image_config.get('convert_to_pil', False):
                img = Image.fromarray(img)
        else:
            img = Image.open(image_path)
        
        # Convert to grayscale if specified
        if image_config.get('grayscale', False):
            img = img.convert('L') if hasattr(img, 'convert') else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert to tensor
        if image_config.get('to_tensor', True):
            img = ToTensor()(img)
            
            # Binarize if specified
            if image_config.get('binarize', False):
                img = (img > image_config.get('binarize_threshold', 0)).float()
        
        if self.transform:
            img = self.transform(img)
            
        return img
    
    def _load_text(self, text_path: str):
        """Load text file"""
        encoding = self.config.get('text_config', {}).get('encoding', 'utf-8')
        with open(text_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def _load_audio(self, audio_path: str):
        """Load audio file (placeholder - requires audio library)"""
        # This would require librosa or similar
        raise NotImplementedError("Audio loading requires additional dependencies")

class SerializedDataset(BaseDataset):
    """Dataset for serialized data (.pth, .pkl, etc.)"""
    
    def __init__(self, config: Dict[str, Any], data_path: str):
        super().__init__(config)
        self.data_path = data_path
        self.serialization_format = config.get('format', 'torch')
        
        self._load_and_split_data()
    
    def _load_and_split_data(self):
        """Load serialized data and create splits using safe formats only.

        Supported formats:
        - safetensors: expects tensor keys (default 'features', 'targets')
        - hdf5: expects dataset keys provided via config ('features_key', 'targets_key')
        - parquet: expects columns provided via config ('features_key', 'targets_key') or defaults

        Deprecated/disabled formats for security: pickle, raw torch .pt/.pth (use safetensors instead).
        """
        fmt = self.serialization_format.lower()
        structure = self.config.get('structure', 'list_of_tuples')
        features_key = self.config.get('features_key', 'features')
        targets_key = self.config.get('targets_key', 'targets')

        if fmt in ('pt', 'pth', 'torch'):
            raise RuntimeError("Loading raw PyTorch checkpoint files is disabled. Please export as safetensors, hdf5, or parquet.")
        if fmt in ('pkl', 'pickle'):
            raise RuntimeError("Loading datasets via pickle is disabled for security. Use safetensors, hdf5, or parquet.")

        data = None

        if fmt == 'safetensors':
           
            tensors = st_load(self.data_path, device='cpu')
            if features_key not in tensors or targets_key not in tensors:
                raise KeyError(f"safetensors file must contain '{features_key}' and '{targets_key}' keys")
            feats = tensors[features_key]
            targs = tensors[targets_key]
            if structure == 'list_of_tuples':
                n = feats.shape[0]
                if targs.shape[0] != n:
                    raise ValueError("features and targets must have the same first dimension")
                data = [(feats[i], targs[i]) for i in range(n)]
            elif structure in ('dict', 'separate_tensors'):
                self.dataset = {features_key: feats, targets_key: targs}
            else:
                raise ValueError(f"Unsupported structure '{structure}' for safetensors")

        elif fmt == 'hdf5':
            with h5py.File(self.data_path, 'r') as f:
                if features_key not in f or targets_key not in f:
                    raise KeyError(f"HDF5 file must contain '{features_key}' and '{targets_key}' datasets")
                feats = f[features_key][...]
                targs = f[targets_key][...]
            if structure == 'list_of_tuples':
                if len(feats) != len(targs):
                    raise ValueError("features and targets must have the same length")
                data = list(zip(feats, targs))
            elif structure in ('dict', 'separate_tensors'):
                self.dataset = {features_key: feats, targets_key: targs}
            else:
                raise ValueError(f"Unsupported structure '{structure}' for hdf5")

        elif fmt == 'parquet':
            # Read with pandas; require feature/target column names
            df = pd.read_parquet(self.data_path)
            if features_key not in df.columns or targets_key not in df.columns:
                raise KeyError(f"Parquet file must contain columns '{features_key}' and '{targets_key}'")
            if structure == 'list_of_tuples':
                data = list(zip(df[features_key].to_list(), df[targets_key].to_list()))
            elif structure in ('dict', 'separate_tensors'):
                self.dataset = {features_key: df[features_key].to_numpy(), targets_key: df[targets_key].to_numpy()}
            else:
                raise ValueError(f"Unsupported structure '{structure}' for parquet")

        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")
        
        # Handle different dataset structures
        structure = self.config.get('structure', 'list_of_tuples')
        
        if structure == 'list_of_tuples':
            # Dataset is a list of (input, target) tuples; if not yet built, convert now
            if data is None:
                # Convert dict of arrays/tensors into list of tuples
                if isinstance(self.dataset, dict):
                    feats = self.dataset.get(features_key)
                    targs = self.dataset.get(targets_key)
                    if feats is None or targs is None:
                        raise KeyError("Missing features/targets in dataset for list_of_tuples structure")
                    n = len(feats)
                    if len(targs) != n:
                        raise ValueError("features and targets must have the same length")
                    data = [(feats[i], targs[i]) for i in range(n)]
                else:
                    data = self.dataset
            # else: data already prepared above
        elif structure == 'dict':
            # Dataset is a dictionary with 'data' and 'targets' keys
            data = list(zip(self.dataset['data'], self.dataset['targets']))
        elif structure == 'separate_tensors':
            # Dataset has separate feature and target tensors
            features = self.dataset[features_key]
            targets = self.dataset[targets_key]
            data = list(zip(features, targets))
        
        # Create splits
        splits = self._create_splits(data)
        
        # Create dataset objects for each split
        for split_name, split_data in splits.items():
            self.splits_data[split_name] = SerializedSplitDataset(
                split_data,
                transform=self.transform
            )

class SerializedSplitDataset(Dataset):
    """Individual dataset for serialized data splits"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if isinstance(sample, tuple) and len(sample) == 2:
            input_data, target = sample
            
            # Apply transforms if specified
            if self.transform:
                input_data = self.transform(input_data)
            
            return input_data, target
        else:
            return sample