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
from typing import Any, Dict, List, Union
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

# --- JSON serialization helper ---
def _to_json_safe(obj: Any):
    """Recursively convert NumPy/Torch scalars/arrays to Python-native types for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    # Fallback: try float conversion, else string
    try:
        return float(obj)
    except Exception:
        return str(obj)

# --- Helpers for segmentation/regression etc. ---
def dice_score_np(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6, threshold: float = 0.5) -> float:
    p = (y_pred > threshold).astype(np.uint8) * 255
    t = (y_true > threshold).astype(np.uint8) * 255
    inter = (p * t).sum()
    return float((2.0 * inter) / (p.sum() + t.sum() + eps))

def jaccard_index_np(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6, threshold: float = 0.5) -> float:
    p = (y_pred > threshold).astype(np.uint8) * 255
    t = (y_true > threshold).astype(np.uint8) * 255
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float(inter / (union + eps))

def hausdorff_distance_np(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    try:
        from scipy.spatial.distance import directed_hausdorff
    except Exception:
        return float("nan")  # scipy not available
    pred_pts = np.argwhere(y_pred > threshold)
    true_pts = np.argwhere(y_true > threshold)
    if len(pred_pts) == 0 or len(true_pts) == 0:
        return float("inf")
    return float(max(directed_hausdorff(pred_pts, true_pts)[0],
                     directed_hausdorff(true_pts, pred_pts)[0]))

# --- Metric registry format:
#   "metric_name": {
#       "fn": callable(y_pred, y_true, params, meta) -> scalar|array|tuple|string,
#       "output": "scalar"|"plot"|"text",
#       "requires_proba": bool (if True, uses probability/score column instead of argmax)
#   }
# meta passed to fn includes {"is_binary": bool, "n_classes": int, "task": str}
# ---
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Classification - scalar
    "accuracy": {
        "fn": lambda yp, yt, p, m: accuracy_score(yt, np.argmax(yp, axis=1) if yp.ndim > 1 else (yp > 0.5).astype(int)),
        "output": "scalar",
        "requires_proba": False
    },
    "f1_score": {
        "fn": lambda yp, yt, p, m: f1_score(
            yt,
            np.argmax(yp, axis=1) if yp.ndim > 1 else (yp > 0.5).astype(int),
            **(p or {"average": ("binary" if m["is_binary"] else "macro")})
        ),
        "output": "scalar",
        "requires_proba": False
    },
    "precision": {
        "fn": lambda yp, yt, p, m: precision_score(
            yt,
            np.argmax(yp, axis=1) if yp.ndim > 1 else (yp > 0.5).astype(int),
            **(p or {"average": ("binary" if m["is_binary"] else "macro")})
        ),
        "output": "scalar",
        "requires_proba": False
    },
    "recall": {
        "fn": lambda yp, yt, p, m: recall_score(
            yt,
            np.argmax(yp, axis=1) if yp.ndim > 1 else (yp > 0.5).astype(int),
            **(p or {"average": ("binary" if m["is_binary"] else "macro")})
        ),
        "output": "scalar",
        "requires_proba": False
    },
    "roc_auc": {
        # for binary: expects score/prob for positive class; for multiclass sklearn supports multi_class param (ovr/ovo) if provided via params
        "fn": lambda yp, yt, p, m: (
            float(roc_auc_score(yt, yp[:, 1], **(p or {}))) if (yp.ndim > 1 and yp.shape[1] > 1 and m["is_binary"])
            else float(roc_auc_score(yt, yp[:, 1], **(p or {}))) if (yp.ndim > 1 and yp.shape[1] == 2)
            else float(roc_auc_score(yt, yp, **(p or {}))) if yp.ndim == 1
            else (float(roc_auc_score(yt, yp, **(p or {}))) if not m["is_binary"] else float("nan"))
        ),
        "output": "scalar",
        "requires_proba": True
    },

    # Classification - plot/text
    "confusion_matrix": {
        "fn": lambda yp, yt, p, m: confusion_matrix(yt, np.argmax(yp, axis=1) if yp.ndim>1 else (yp>0.5).astype(int)),
        "output": "plot",
        "requires_proba": False,
        "filename": "confusion_matrix.png"
    },
    "classification_report": {
        "fn": lambda yp, yt, p, m: classification_report(yt, np.argmax(yp, axis=1) if yp.ndim>1 else (yp>0.5).astype(int)),
        "output": "text",
        "requires_proba": False,
        "filename": "classification_report.txt"
    },
    "precision_recall_curve": {
        "fn": lambda yp, yt, p, m: precision_recall_curve(yt, (yp[:, 1] if (yp.ndim>1 and yp.shape[1]>1) else yp).ravel()),
        "output": "plot",
        "requires_proba": True,
        "filename": "precision_recall_curve.png"
    },
    "roc_curve": {
        "fn": lambda yp, yt, p, m: roc_curve(yt, (yp[:, 1] if (yp.ndim>1 and yp.shape[1]>1) else yp).ravel()),
        "output": "plot",
        "requires_proba": True,
        "filename": "roc_curve.png"
    },

    # Segmentation
    "dice_score": {
        "fn": lambda yp, yt, p, m: dice_score_np(yp, yt, threshold=p.get("threshold", 0.5)),
        "output": "scalar",
        "requires_proba": False
    },
    "jaccard_index": {
        "fn": lambda yp, yt, p, m: jaccard_index_np(yp, yt, threshold=p.get("threshold", 0.5)),
        "output": "scalar",
        "requires_proba": False
    },
    "hausdorff_distance": {
        "fn": lambda yp, yt, p, m: hausdorff_distance_np(yp, yt, threshold=p.get("threshold", 0.5)),
        "output": "scalar",
        "requires_proba": False
    },

    # Regression
    "mse": {
        "fn": lambda yp, yt, p, m: float(mean_squared_error(yt, yp)),
        "output": "scalar",
        "requires_proba": False
    },
    "mae": {
        "fn": lambda yp, yt, p, m: float(mean_absolute_error(yt, yp)),
        "output": "scalar",
        "requires_proba": False
    },
    "rmse": {
        "fn": lambda yp, yt, p, m: float(mean_squared_error(yt, yp, squared=False)),
        "output": "scalar",
        "requires_proba": False
    },
    "r2_score": {
        "fn": lambda yp, yt, p, m: float(r2_score(yt, yp)),
        "output": "scalar",
        "requires_proba": False
    }
}

# --- Utility: parse metrics config: allow string or dict {"name":.., "params":{...}} ---
def parse_metrics_config(metrics_config: Union[List[Any], None]) -> List[Dict[str, Any]]:
    if not metrics_config:
        return []
    parsed = []
    for entry in metrics_config:
        if isinstance(entry, str):
            parsed.append({"name": entry, "params": None})
        elif isinstance(entry, dict):
            parsed.append({"name": entry.get("name"), "params": entry.get("params", None)})
        else:
            raise ValueError("Metric config entries must be either str or dict")
    return parsed


def compute_metrics(preds_list, targets_list, test_loss=None, config=None):
    metrics = parse_metrics_config(config.get("metrics", []))
    task_type = config.get("task_type", "")
    save_path = config.get("paths", {}).get("trained_model_output_path", "")
    # n_pred_samples = config.get("n_pred_samples", 0)
    threshold = config.get("threshold", 0.5) if config else 0.5

    if len(preds_list) == 0:
        raise ValueError("Predictions on test set are empty. Please check the test loader.")

    # y_pred_all = torch.cat(preds_list, dim=0).numpy()
    # y_true_all = torch.cat(targets_list, dim=0).numpy() if len(targets_list) else None
    y_pred_all = np.array(preds_list)
    y_true_all = np.array(targets_list) if len(targets_list) else None

    # ------------------------
    # Auto-choose default metrics if none provided
    # ------------------------
    if not metrics:
        if task_type == "classification" and y_true_all is not None:
            n_classes = len(np.unique(y_true_all))
            if n_classes == 2:
                metrics = [{"name": "classification_report"}, {"name": "roc_auc"}]
            else:
                metrics = [{"name": "classification_report"}]   
        elif task_type == "segmentation":
            metrics = [{"name": "dice_score"}, {"name": "jaccard_index"}]
        elif task_type == "regression":
            metrics = [{"name": "mse"}, {"name": "mae"}, {"name": "r2_score"}]

    # meta info passed to metric fns
    meta = {}
    if y_true_all is not None and task_type == "classification":
        unique = np.unique(y_true_all)
        meta["n_classes"] = int(unique.size)
        meta["is_binary"] = (unique.size == 2)
    else:
        meta["n_classes"] = None
        meta["is_binary"] = False

    # ------------------------
    # Metric computation
    # ------------------------
    if test_loss is not None:
        numeric_metrics = {"test_loss": test_loss} 
    else:
        numeric_metrics = {}

    if y_true_all is not None:
        n_classes = len(np.unique(y_true_all)) if task_type == "classification" else None
        is_binary = (n_classes == 2)

    for m in metrics:
        name = m["name"]
        params = m.get("params", None)
        entry = METRIC_REGISTRY.get(name)
        if entry is None:
            numeric_metrics[name] = f"Metric {name} not implemented. Please raise an issue on GitHub."
            continue
        try:
            result = entry["fn"](y_pred_all, y_true_all, params, {"is_binary": meta["is_binary"], "n_classes": meta["n_classes"], "task": task_type})

            # Route outputs by declared type
            if entry["output"] == "scalar":
                # ensure JSON serializable float
                numeric_metrics[name] = float(result) if (isinstance(result, (int, float, np.floating, np.integer))) else result

            elif entry["output"] == "text":
                txt = str(result)
                fname = entry.get("filename", f"{name}.txt")
                if save_path:
                    with open(os.path.join(save_path, fname), "w") as f:
                        f.write(txt)
                # textual outputs not included in numeric JSON

            elif entry["output"] == "plot":
                # Expect result to be either array-like (matrix) or tuple for curve (x,y) or (precision,recall,_)
                fname = entry.get("filename", f"{name}.png")
                if save_path:
                    plt.figure()
                    # confusion matrix -> 2D array
                    if isinstance(result, (list, np.ndarray)) and np.asarray(result).ndim == 2:
                        arr = np.asarray(result)
                        plt.imshow(arr, interpolation='nearest', cmap='Blues')
                        plt.colorbar()
                        plt.title(name.replace("_", " ").title())
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                    # curve -> tuple (x,y,maybe thresholds)
                    elif isinstance(result, (tuple, list)) and len(result) >= 2:
                        x, y = result[0], result[1]
                        plt.plot(x, y)
                        plt.title(name.replace("_", " ").title())
                        plt.xlabel("x")
                        plt.ylabel("y")
                    else:
                        # fallback: try plotting 1D
                        arr = np.asarray(result)
                        if arr.ndim == 1:
                            plt.plot(arr)
                            plt.title(name.replace("_", " ").title())
                        else:
                            plt.text(0.1, 0.5, "Cannot plot result", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, fname))
                    plt.close()

        except Exception as e:
            numeric_metrics[name] = f"Failed: {e}"

    # Save metrics
    if save_path and len(numeric_metrics) > 0:
        with open(os.path.join(save_path, "evaluation_metrics.json"), "w") as f:
            json.dump(numeric_metrics, f, indent=4, default=_to_json_safe)


    # TO EXPLORE: Can sample predictions be saved without breaking privacy constraints?

    """
    # Save sample predictions
    if save_path and n_pred_samples > 0 and y_true_all is not None:
        nsave = min(n_pred_samples, len(y_pred_all))
        
        # Pre-compute common paths and conversions
        if task_type == "segmentation":
            for i in range(nsave):
                pred_img = (y_pred_all[i] > threshold).astype(np.uint8) * 255
                
                # plt.imsave(os.path.join(save_path, f"pred_{i+1}_binarized.png"), pred_img, cmap="gray", vmin=0, vmax=255)
                plt.imsave(os.path.join(save_path, f"pred_{i+1}.png"), y_pred_all[i], cmap="gray", vmin=0, vmax=255)
                # plt.imsave(os.path.join(save_path, f"mask_{i+1}.png"), true_img, cmap="gray")
        
        elif task_type == "classification":
            is_multiclass = y_pred_all.ndim > 1 and y_pred_all.shape[1] > 1
            preds = y_pred_all[:nsave]
            trues = y_true_all[:nsave]
            
            for i in range(nsave):
                if is_multiclass:
                    pred_data = {
                        "pred_label": int(np.argmax(preds[i])),
                        "scores": preds[i].tolist(),
                        "true": trues[i].tolist() if hasattr(trues[i], "tolist") else int(trues[i])
                    }
                else:
                    pred_data = {
                        "pred_label": int((preds[i] > 0.5).astype(int)),
                        "scores": float(preds[i].ravel()[0]),
                        "true": int(trues[i]) if hasattr(trues[i], "__iter__") and not isinstance(trues[i], str) else trues[i]
                    }
                with open(os.path.join(save_path, f"pred_{i+1}.txt"), "w") as f:
                    json.dump(pred_data, f, default=_to_json_safe)
                    
        elif task_type == "regression":
            preds = y_pred_all[:nsave]
            trues = y_true_all[:nsave]
            
            for i in range(nsave):
                pred_data = {
                    "pred": float(preds[i].ravel()[0]) if np.asarray(preds[i]).size==1 else preds[i].tolist(),
                    "true": float(trues[i].ravel()[0]) if np.asarray(trues[i]).size==1 else trues[i].tolist()
                }
                with open(os.path.join(save_path, f"pred_{i+1}.txt"), "w") as f:
                    json.dump(pred_data, f, default=_to_json_safe)
    """

    return numeric_metrics