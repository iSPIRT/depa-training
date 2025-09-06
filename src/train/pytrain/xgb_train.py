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
import torch

from .task_base import TaskBase
from .utilities.dataset_constructor import create_dataset
from .utilities.eval_tools import compute_metrics

import xgboost as xgb
from .utilities.dp_xgboost import DPXGBoost
from sklearn.metrics import mean_squared_error

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
    """Train (DP-)XGBoost models.

    Expected config snippet:
    {
        "task_type": "classification" | "regression",
        "dataset_config": {...},
        "paths": {"input_dataset_path": "/path", "trained_model_output_path": "/out"},
        "model_config": {
            "n_estimators": 300, 
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "num_class": 2,
            "booster_params": {"tree_method": "hist", ...}  # optional, passed to underlying booster
        },
        "is_private": true,
        "privacy_params": {"mechanism": "gaussian", "epsilon": 2.0, "delta": 1e-5, "clip_value": 1.0}  # used by some dp-xgboost forks
    }
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

        self.model = None

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

    def load_model(self):
        xgb_params = self.config["model_config"]["booster_params"]
        self.model = DPXGBoost(xgb_params, privacy_params=self.privacy_config)

    def train(self):
        num_boost_round = self.config["model_config"]["num_boost_round"]
        self.model.fit(X=self.X_train, y=self.y_train, num_boost_round=num_boost_round)
        if self.is_private:
            eps_string = f"Epsilon: {self.privacy_config['epsilon']}"
        else:
            eps_string = "Non DP"
        print(f"Trained Gradient Boosting model with {num_boost_round} boosting rounds | {eps_string}")

    def save_model(self):
        save_path = os.path.join(self.paths["trained_model_output_path"], "trained_model.json")
        self.model.save_model(save_path)
        print(f"Saved model to {self.paths['trained_model_output_path']}")

    def inference_and_eval(self):
        if self.X_test is None:
            print("No test split provided; skipping evaluation.")
            return

        preds_list = []
        targets_list = []
        non_dp_preds_list = []

        if self.is_private:
            preds_list = self.model.predict(X=self.X_test, dp=True)
            non_dp_preds_list = self.model.predict(X=self.X_test, dp=False)
        else:
            preds_list = self.model.predict(X=self.X_test, dp=False)

        targets_list = self.y_test

        numeric_metrics = compute_metrics(preds_list, targets_list, None, self.config)
        print(f"Evaluation Metrics: {numeric_metrics}")

        if self.is_private:
            os.makedirs(os.path.join(self.paths["trained_model_output_path"], "non_dp"), exist_ok=True)
            self.config["paths"]["trained_model_output_path"] = os.path.join(self.paths["trained_model_output_path"], "non_dp")
            non_dp_numeric_metrics = compute_metrics(non_dp_preds_list, targets_list, None, self.config)
            print(f"Non-DP Evaluation Metrics: {non_dp_numeric_metrics}")

    def execute(self, config: Dict[str, Any]):
        try:
            self.init(config)
            self.load_data()
            self.load_model()
            self.train()
            self.save_model()
            self.inference_and_eval()
            print("CCR Training complete!\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e


