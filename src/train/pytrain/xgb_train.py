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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
import torch

from .task_base import TaskBase
from .utilities.dataset_constructor import create_dataset
from .utilities.eval_tools import compute_metrics

import xgboost as xgb
# Sarus Tech dp-xgboost library (use their xgb namespace)
import dp_xgboost as dp_xgb

def _to_numpy(array_like: Any) -> np.ndarray:
    try:
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(array_like, "to_numpy"):
        return array_like.to_numpy()
    return np.asarray(array_like)


def _extract_xy_from_split(split_ds: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if hasattr(split_ds, "features"):
        X = _to_numpy(getattr(split_ds, "features"))
        y = getattr(split_ds, "targets")
        y_np = _to_numpy(y) if y is not None else None
        if y_np is not None and y_np.ndim > 1 and y_np.shape[1] == 1:
            y_np = y_np.ravel()
        return X, y_np

    features_list = []
    targets_list = []
    has_target = None
    for i in range(len(split_ds)):
        item = split_ds[i]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            x, y = item
            features_list.append(_to_numpy(x))
            targets_list.append(_to_numpy(y))
            has_target = True
        else:
            features_list.append(_to_numpy(item))
            if has_target is None:
                has_target = False

    X = np.vstack([x.reshape(1, -1) if x.ndim == 1 else x for x in features_list]) if len(features_list) else np.empty((0,))
    y = None
    if has_target:
        y_arr = np.asarray(targets_list)
        if y_arr.ndim > 1 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        y = y_arr
    return X, y


class Train_XGB(TaskBase):
    """Train (DP-)XGBoost models using Sarus Tech's dp-xgboost library.
    - Uses xgb.train() API (not sklearn XGBClassifier/XGBRegressor)
    - Enables DP via tree_method='approxDP' parameter
    - Requires DMatrix with feature_min/feature_max bounds
    - Uses dp_epsilon_per_tree = total_epsilon / n_estimators
    - Automatically handles class imbalance with scale_pos_weight

    Expected config snippet:
    {
        "task_type": "classification" | "regression",
        "dataset_config": {...},
        "paths": {"input_dataset_path": "...", "trained_model_output_path": "..."},
        "is_private": true,
        "privacy_params": {
            "epsilon": 4.0,  # Total privacy budget (split across trees)
            "delta": 1e-5    # Privacy parameter (typically 1/n^2)
        },
        "model_config": {
            "n_estimators": 250,            # Number of boosting rounds
            "max_depth": 6,                 # Maximum tree depth
            "learning_rate": 0.05,          # Learning rate (eta)
            "l2_regularization": 1.0,       # L2 regularization (lambda)
            "min_child_weight": 1.0,        # Minimum sum of instance weight
            "subsample": 1.0,               # Subsample ratio (affects privacy)
            "objective": "binary:logistic", # Loss function
            "seed": 42,                     # Random seed for reproducibility
            "scale_pos_weight": None        # Optional: override auto-calculated value
        }
    }
    
    Implementation Details:
    - Total epsilon is divided: dp_epsilon_per_tree = epsilon / n_estimators
    - Actual epsilon spent: n_trees * log(1 + subsample*(e^eps_per_tree - 1))
    - When is_private=True, both DP and non-DP models are trained for comparison
    - Feature bounds are automatically computed from training data
    - scale_pos_weight is automatically calculated for imbalanced classes
    
    Reference: https://github.com/Sarus-Tech/dp-xgboost
    """

    def init(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = config.get("task_type")
        self.paths = config.get("paths", {})

        self.is_private = bool(config["is_private"])
        self.privacy_config = config["privacy_params"] if self.is_private else None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.bst = None  # DP booster
        self.non_dp_bst = None  # Non-DP baseline booster
        
        # Feature bounds required for DP
        self.feature_min = None
        self.feature_max = None

    def load_data(self):
        dataset_cfg = self.config.get("dataset_config", {})
        input_path = self.paths.get("input_dataset_path")
        all_splits = create_dataset(dataset_cfg, input_path)

        if "train" not in all_splits:
            raise ValueError("Dataset must provide at least a 'train' split")

        self.X_train, self.y_train = _extract_xy_from_split(all_splits["train"])
        if "val" in all_splits:
            self.X_val, self.y_val = _extract_xy_from_split(all_splits["val"])
        if "test" in all_splits:
            self.X_test, self.y_test = _extract_xy_from_split(all_splits["test"])

        print(f"Loaded dataset splits | train: {self.X_train.shape} | val: {None if self.X_val is None else self.X_val.shape} | test: {None if self.X_test is None else self.X_test.shape}")
        
        # Calculate scale_pos_weight for class imbalance (classification only)
        self.scale_pos_weight = None
        if self.task_type == "classification" and self.y_train is not None:
            unique, counts = np.unique(self.y_train, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            # For binary classification, calculate scale_pos_weight
            if len(unique) == 2 and 0 in class_counts and 1 in class_counts:
                # scale_pos_weight = count(negative) / count(positive)
                self.scale_pos_weight = class_counts[0] / class_counts[1]
                print(f"Class imbalance detected: {class_counts}")
                print(f"Calculated scale_pos_weight: {self.scale_pos_weight:.4f}")
        
        # Compute feature bounds (required for DP)
        if self.is_private:
            n_features = self.X_train.shape[1]
            self.feature_min = [float(np.min(self.X_train[:, i])) for i in range(n_features)]
            self.feature_max = [float(np.max(self.X_train[:, i])) for i in range(n_features)]
            print(f"Feature bounds for DP: min: {np.mean(self.feature_min):.4f}, max: {np.mean(self.feature_max):.4f}")

    def train(self):
        """Train the model using Sarus Tech's DP-XGBoost with xgb.train() API."""
        model_config = self.config["model_config"]
        
        # model parameters
        n_estimators = model_config.get("n_estimators", 100)
        max_depth = model_config.get("max_depth", 6)
        learning_rate = model_config.get("learning_rate", 0.1)
        l2_reg = model_config.get("l2_regularization", 1.0)
        min_child_weight = model_config.get("min_child_weight", 1.0)
        objective = model_config.get("objective", "binary:logistic" if self.task_type == "classification" else "reg:squarederror")
        default_base_score = 0.5 if self.task_type == "classification" else 0.0
        base_score = model_config.get("base_score", default_base_score)
        subsample = model_config.get("subsample", 1.0)
        seed = model_config.get("seed", 42)  # Random seed for reproducibility
        
        # Handle class imbalance with scale_pos_weight (can be overridden in config)
        scale_pos_weight = model_config.get("scale_pos_weight", self.scale_pos_weight)
        
        if self.is_private:
            epsilon = self.privacy_config.get("epsilon", 1.0)
            delta = self.privacy_config.get("delta", 1e-5)
            
            # Calculate per-tree epsilon budget (important for composition)
            dp_epsilon_per_tree = epsilon / n_estimators
            # Calculate total budget spent using advanced composition
            total_budget_spent = n_estimators * np.log(1 + subsample * (np.exp(dp_epsilon_per_tree) - 1))
            
            print(f"\n[DEBUG] DP Model Configuration:")
            print(f"  - n_estimators: {n_estimators}")
            print(f"  - max_depth: {max_depth}")
            print(f"  - learning_rate: {learning_rate}")
            print(f"  - lambda (l2_reg): {l2_reg}")
            print(f"  - min_child_weight: {min_child_weight}")
            print(f"  - subsample: {subsample}")
            print(f"  - objective: {objective}")
            print(f"  - epsilon (total): {epsilon}, delta: {delta}")
            print(f"  - dp_epsilon_per_tree: {dp_epsilon_per_tree:.6f}")
            print(f"  - Total epsilon spent (with composition): {total_budget_spent:.4f}")
            print(f"  - tree_method: approxDP (REQUIRED for DP)")
            print(f"  - dp_xgboost version: {getattr(dp_xgb, '__version__', 'unknown')}")
            
            # Create DMatrix with feature bounds (REQUIRED for DP)
            dtrain = dp_xgb.DMatrix(
                self.X_train, 
                label=self.y_train,
                feature_min=self.feature_min,
                feature_max=self.feature_max
            )
            
            # DP parameters for xgb.train()
            params_dp = {
                'objective': objective,
                'tree_method': 'approxDP',  # CRITICAL: This enables DP
                'dp_epsilon_per_tree': dp_epsilon_per_tree,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'lambda': l2_reg,
                'base_score': base_score,
                'subsample': subsample,
                'min_child_weight': min_child_weight,
                'seed': seed,  # Set seed for reproducible DP noise
            }
            
            # Add scale_pos_weight if available (for class imbalance)
            if scale_pos_weight is not None:
                params_dp['scale_pos_weight'] = scale_pos_weight
                print(f"  - scale_pos_weight: {scale_pos_weight:.4f} (handling class imbalance)")
            
            # Train DP model
            print(f"\nTraining DP model with ε={epsilon:.2f}")
            self.bst = dp_xgb.train(params_dp, dtrain, num_boost_round=n_estimators)
            print("DP model trained")
            
            # Train non-DP baseline for comparison
            print("\nTraining non-DP baseline model for comparison...")
            
            # non-DP parameters (use 'approx' tree_method, not 'approxDP')
            params_non_dp = {
                'objective': objective,
                'tree_method': 'approx',  # Standard (non-DP) method
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'lambda': l2_reg,
                'base_score': base_score,
                'subsample': subsample,
                'min_child_weight': min_child_weight,
                'seed': seed,  # Same seed for fair comparison
            }
            
            # Add scale_pos_weight if available (for class imbalance)
            if scale_pos_weight is not None:
                params_non_dp['scale_pos_weight'] = scale_pos_weight
            
            # Create regular DMatrix (no feature bounds needed)
            dtrain_non_dp = xgb.DMatrix(self.X_train, label=self.y_train)
            # Train non-DP model
            self.non_dp_bst = xgb.train(params_non_dp, dtrain_non_dp, num_boost_round=n_estimators)
            print("Non-DP baseline model trained")
            
        else:
            # Standard non-DP training
            params_non_dp = {
                'objective': objective,
                'tree_method': 'approx',
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'lambda': l2_reg,
                'base_score': base_score,
                'subsample': subsample,
                'min_child_weight': min_child_weight,
                'seed': seed,  # Set seed for reproducibility
            }
            
            # Add scale_pos_weight if available (for class imbalance)
            if scale_pos_weight is not None:
                params_non_dp['scale_pos_weight'] = scale_pos_weight
                print(f"Using scale_pos_weight: {scale_pos_weight:.4f} (handling class imbalance)")
            
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            self.bst = xgb.train(params_non_dp, dtrain, num_boost_round=n_estimators)
            print(f"Trained standard XGBoost model with {n_estimators} boosting rounds")

    def save_model(self):
        save_path = os.path.join(self.paths["trained_model_output_path"], "trained_model.json")
        # Save the booster directly
        if self.bst is not None:
            self.bst.save_model(save_path)
            print(f"Saved model to {self.paths['trained_model_output_path']}")


    def inference_and_eval(self):
        """Run inference and evaluation on test set."""
        if self.X_test is None:
            print("No test split provided; skipping evaluation.")
            return

        # Create DMatrix for test set
        if self.is_private:
            dtest = dp_xgb.DMatrix(
                self.X_test,
                feature_min=self.feature_min,
                feature_max=self.feature_max
            )
        else:
            dtest = xgb.DMatrix(self.X_test)
        
        # Get predictions from DP model
        preds_raw = self.bst.predict(dtest)
        
        # For classification, convert to binary predictions
        if self.task_type == "classification":
            preds_list = (preds_raw > 0.5).astype(int)
        else:
            preds_list = preds_raw
        
        targets_list = self.y_test

        print(f"\n{'='*60}")
        print(f"DP Model Evaluation (ε={self.privacy_config['epsilon']})" if self.is_private else "Model Evaluation")
        print(f"{'='*60}")
        numeric_metrics = compute_metrics(preds_list, targets_list, None, self.config)
        print(f"Evaluation Metrics: {numeric_metrics}")

        # Evaluate non-DP baseline model if available
        if self.is_private and self.non_dp_bst is not None:
            dtest_non_dp = xgb.DMatrix(self.X_test)
            non_dp_preds_raw = self.non_dp_bst.predict(dtest_non_dp)
            
            # For classification, convert to binary predictions
            if self.task_type == "classification":
                non_dp_preds_list = (non_dp_preds_raw > 0.5).astype(int)
            else:
                non_dp_preds_list = non_dp_preds_raw
            
            original_output_path = self.paths["trained_model_output_path"]
            non_dp_output_path = os.path.join(original_output_path, "non_dp")
            os.makedirs(non_dp_output_path, exist_ok=True)
            self.config["paths"]["trained_model_output_path"] = non_dp_output_path
            
            print(f"\n{'='*60}")
            print(f"Non-DP Baseline Model Evaluation")
            print(f"{'='*60}")
            non_dp_numeric_metrics = compute_metrics(non_dp_preds_list, targets_list, None, self.config)
            print(f"Non-DP Evaluation Metrics: {non_dp_numeric_metrics}")
            self.config["paths"]["trained_model_output_path"] = original_output_path
            
            # Print comparison
            print(f"\n{'='*60}")
            print(f"Privacy-Utility Trade-off Summary")
            print(f"{'='*60}")
            if 'accuracy' in numeric_metrics and 'accuracy' in non_dp_numeric_metrics:
                utility_loss = non_dp_numeric_metrics['accuracy'] - numeric_metrics['accuracy']
                print(f"Accuracy Loss due to DP: {utility_loss:.4f} ({utility_loss*100:.2f}%)")
            if 'roc_auc' in numeric_metrics and 'roc_auc' in non_dp_numeric_metrics:
                auc_loss = non_dp_numeric_metrics['roc_auc'] - numeric_metrics['roc_auc']
                print(f"ROC-AUC Loss due to DP: {auc_loss:.4f} ({auc_loss*100:.2f}%)")
            print(f"Privacy Budget: ε={self.privacy_config['epsilon']}, δ={self.privacy_config['delta']}")
            print(f"{'='*60}\n")

    def execute(self, config: Dict[str, Any]):
        try:
            self.init(config)
            self.load_data()
            self.train()
            self.save_model()
            self.inference_and_eval()
            print("CCR Training complete!\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e



