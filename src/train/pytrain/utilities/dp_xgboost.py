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

from typing import Any, Dict, Optional, Sequence, Tuple
import json
import math
import logging

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DPXGBoost:
    def __init__(self, xgb_params, privacy_params = None):
        """
        xgb_params: config passed to xgb.train
        privacy_params: optional config.
            mechanism: 'laplace' or 'gaussian'
            epsilon: float (required)
            delta: float (required when gaussian)
            clip_value: float (required)
        """
        self.xgb_params = xgb_params
        self.privacy_params = privacy_params
        self.privacy_enabled = True if privacy_params is not None else False

        # Validate privacy params only if enabled
        if self.privacy_enabled:
            mech = self.privacy_params.get('mechanism', 'gaussian')
            if mech not in ('laplace', 'gaussian'):
                raise ValueError("mechanism must be 'laplace' or 'gaussian'")
            if 'epsilon' not in self.privacy_params:
                raise ValueError('epsilon is required in privacy_params when privacy enabled')
            if mech == 'gaussian' and 'delta' not in self.privacy_params:
                raise ValueError('delta is required for gaussian mechanism')
            if 'clip_value' not in self.privacy_params:
                raise ValueError('clip_value is required in privacy_params')

        # bookkeeping
        self.bst: Optional[xgb.Booster] = None
        self._noisy_leaf_values: Optional[Dict[Tuple[int, int], float]] = None
        self._orig_leaf_values: Optional[Dict[Tuple[int, int], float]] = None
        self._trained_num_trees: Optional[int] = None

    # Delegation: let wrapper expose Booster attributes once a booster exists
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self.bst is not None and hasattr(self.bst, name):
            return getattr(self.bst, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


    def fit(self,
            X: Any = None,
            y: Any = None,
            num_boost_round: int = 100,
            dtrain: Optional[xgb.DMatrix] = None,
            evals: Optional[Sequence[Tuple[xgb.DMatrix, str]]] = None,
            **train_kwargs) -> xgb.Booster:
        """
        Train via xgb.train. If privacy enabled, compute noisy leaf mapping.
        """
        if dtrain is None:
            if X is None or y is None:
                raise ValueError("Provide either dtrain or X and y")
            dtrain = xgb.DMatrix(X, label=y)

        if self.privacy_enabled:
            logger.info("Privacy enabled: DP noise will be computed after training.")
        else:
            logger.info("Privacy disabled: running standard non-DP training.")

        # Train using xgboost's train (delegates heavy training to xgboost)
        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=num_boost_round, evals=evals, **train_kwargs)

        # cache number of trees (dump length)
        dump = self.bst.get_dump(dump_format='json')
        self._trained_num_trees = len(dump)
        logger.info("Trained %d trees", self._trained_num_trees)

        # original leaf values
        self._orig_leaf_values = self._extract_leaf_values(self.bst)

        # compute noisy mapping only if enabled, otherwise copy original map
        if self.privacy_enabled:
            self._noisy_leaf_values = self._compute_dp_leaf_noise(self.bst, dtrain)
        else:
            self._noisy_leaf_values = dict(self._orig_leaf_values)

        return self.bst

    def predict(self, X: Any, dp: bool = False, **predict_kwargs) -> np.ndarray:
        """
        If dp=False: delegate to underlying Booster.predict.
        If dp=True:
          - If privacy disabled -> returns same as non-DP predictions (we use noisy_map==orig_map).
          - If privacy enabled -> compute predictions by summing noisy leaf values.
        NOTE: For binary classification with objective 'binary:logistic', booster.predict(dp=False)
        returns probabilities, whereas predict(dp=True) returns the raw margin from base_score +
        sum(eta * noisy_leaf). Convert with sigmoid if you need probability-like output.
        """
        if self.bst is None:
            raise ValueError("Model not trained yet. Call fit(...) first.")

        dmat = xgb.DMatrix(X)
        if not dp:
            return self.bst.predict(dmat, **predict_kwargs)

        # dp prediction (noisy leaf aggregation)
        if self._noisy_leaf_values is None or self._orig_leaf_values is None:
            raise ValueError("No leaf mappings available. Call fit(...) first.")

        leaves = self.bst.predict(dmat, pred_leaf=True)
        n_samples, n_trees = leaves.shape
        if self._trained_num_trees is not None and n_trees != self._trained_num_trees:
            logger.warning("pred_leaf reported %d trees but trained had %d", n_trees, self._trained_num_trees)

        lr = float(self.xgb_params.get("eta", self.xgb_params.get("learning_rate", 0.3)))
        base = float(self.xgb_params.get("base_score", 0.5))

        preds = np.full(n_samples, base, dtype=float)

        for t in range(n_trees):
            # build vector of leaf values for tree t
            tree_vals = np.zeros(n_samples, dtype=float)
            for i in range(n_samples):
                leaf_id = int(leaves[i, t])
                key = (t, leaf_id)
                val = self._noisy_leaf_values.get(key)
                if val is None:
                    val = self._orig_leaf_values.get(key, 0.0)
                tree_vals[i] = val
            preds += lr * tree_vals

        return preds

    def save_model(self, path: str) -> None:
        if self.bst is None:
            raise ValueError("No trained model to save")
        return self.bst.save_model(path)

    def load_model(self, path: str) -> xgb.Booster:
        self.bst = xgb.Booster()
        self.bst.load_model(path)
        dump = self.bst.get_dump(dump_format='json')
        self._trained_num_trees = len(dump)
        self._orig_leaf_values = self._extract_leaf_values(self.bst)
        self._noisy_leaf_values = None
        return self.bst

    def get_booster(self) -> Optional[xgb.Booster]:
        return self.bst

    # ----------------
    # Internal helpers
    # ----------------
    def _extract_leaf_values(self, booster: xgb.Booster) -> Dict[Tuple[int, int], float]:
        """
        Return mapping (tree_idx, nodeid) -> leaf_value.
        Prefer trees_to_dataframe, fallback to JSON dump parsing.
        """
        mapping: Dict[Tuple[int, int], float] = {}
        try:
            df = booster.trees_to_dataframe()
            if "Leaf" in df.columns:
                leaf_rows = df.loc[df["Leaf"].notnull(), ["Tree", "Node", "Leaf"]]
                for _, row in leaf_rows.iterrows():
                    t = int(row["Tree"])
                    nodeid = int(row["Node"])
                    val = float(row["Leaf"])
                    mapping[(t, nodeid)] = val
                if mapping:
                    return mapping
        except Exception:
            logger.debug("trees_to_dataframe failed; falling back to JSON parse", exc_info=True)

        dump = booster.get_dump(dump_format="json")
        for t, tree_json in enumerate(dump):
            try:
                tree = json.loads(tree_json)
            except Exception:
                continue

            def _walk(node):
                if "leaf" in node:
                    nodeid = int(node.get("nodeid", -1))
                    mapping[(t, nodeid)] = float(node["leaf"])
                else:
                    if "children" in node and isinstance(node["children"], list):
                        for c in node["children"]:
                            _walk(c)
                    else:
                        for k in ("yes", "no", "missing"):
                            child = node.get(k)
                            if isinstance(child, dict):
                                _walk(child)

            if isinstance(tree, dict):
                _walk(tree)
            elif isinstance(tree, list):
                for node in tree:
                    _walk(node)

        return mapping

    def _compute_dp_leaf_noise(self, booster: xgb.Booster, dtrain: xgb.DMatrix) -> Dict[Tuple[int, int], float]:
        """
        Compute noisy leaf mapping using a simple per-tree uniform budget split.
        Sensitivity approximated as: clip_value / (leaf_count * min_hessian + l2_reg)
        """
        if self._orig_leaf_values is None:
            self._orig_leaf_values = self._extract_leaf_values(booster)

        leaves = booster.predict(dtrain, pred_leaf=True)
        n_samples, n_trees = leaves.shape
        num_trees = n_trees

        # counts per (tree, leaf)
        leaf_counts: Dict[Tuple[int, int], int] = {}
        for i in range(n_samples):
            for t in range(num_trees):
                leafid = int(leaves[i, t])
                key = (t, leafid)
                leaf_counts[key] = leaf_counts.get(key, 0) + 1

        mech = self.privacy_params.get("mechanism", "gaussian")
        epsilon = float(self.privacy_params["epsilon"])
        delta = float(self.privacy_params.get("delta", 0.0))
        clip = float(self.privacy_params["clip_value"])
        min_hessian = float(self.privacy_params.get("min_hessian", 1.0))
        budget_alloc = self.privacy_params.get("budget_allocation", "uniform")
        privacy_seed = self.privacy_params.get("privacy_seed", None)

        l2_reg = float(self.xgb_params.get("lambda", self.xgb_params.get("reg_lambda", 1.0)))

        if budget_alloc != "uniform":
            logger.warning("Only uniform budget_allocation supported; falling back to uniform")
        eps_per_tree = epsilon / max(1, num_trees)
        delta_per_tree = (delta / max(1, num_trees)) if mech == "gaussian" and delta > 0 else None

        rng = np.random.default_rng(privacy_seed)

        noisy_map: Dict[Tuple[int, int], float] = {}
        for (t, leafid), orig_val in self._orig_leaf_values.items():
            count = leaf_counts.get((t, leafid), 0)
            denom = count * min_hessian + l2_reg
            if denom <= 0:
                denom = 1e-6
            sensitivity = clip / denom

            if mech == "laplace":
                scale = sensitivity / max(1e-12, eps_per_tree)
                noise = rng.laplace(0.0, scale)
            else:
                if delta_per_tree is None or delta_per_tree <= 0:
                    raise ValueError("delta must be positive for gaussian mechanism")
                sigma = math.sqrt(2.0 * math.log(1.25 / delta_per_tree)) * sensitivity / max(1e-12, eps_per_tree)
                noise = rng.normal(0.0, sigma)

            noisy_map[(t, leafid)] = float(orig_val + noise)

        logger.info("Computed noisy leaf mapping for %d trees", num_trees)
        return noisy_map