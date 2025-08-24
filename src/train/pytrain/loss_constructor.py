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


import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.losses
# import kornia.losses
# import torchgan.losses
# import piq

import importlib


class LossComposer:
    def __init__(self, config):
        self.config = config
        self.loss_fn = self._parse_config(config)

    @classmethod
    def load_from_dict(cls, config):
        return cls(config)

    def calculate_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    # ----------------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------------
    def _parse_config(self, config):
        if "class" in config:
            # Atomic loss
            return self._build_atomic_loss(config)
        elif "expression" in config:
            # Composed loss
            return self._build_composed_loss(config)
        else:
            raise ValueError("Invalid loss config: must contain 'class' or 'expression'")

    def _build_atomic_loss(self, config):
      cls_path = config["class"]
      module_name, cls_name = cls_path.rsplit(".", 1)
      module = importlib.import_module(module_name) # Only applicable for approved libraries, included in ci/Dockerfile.train
      obj = getattr(module, cls_name)

      if callable(obj) and not isinstance(obj, type):
          # It's a function (like torch.exp)
          def fn(outputs, targets, cache=None):
              params = config.get("params", {})
              resolved = self._resolve_params(params, outputs, targets, cache)
              return obj(**resolved)
          return fn
      else:
          # It's a class, instantiate
          instance = obj(**config.get("params", {}))
          def fn(outputs, targets, cache=None):
              return instance(outputs, targets)
          return fn

    def _build_composed_loss(self, config):
        expression = config["expression"]
        components_cfg = config.get("components", {})
        variables = config.get("variables", {})
        reduction = config.get("reduction", "mean")

        # Recursively build sub-components
        components = {
            name: self._parse_config(sub_cfg)
            for name, sub_cfg in components_cfg.items()
        }

        def fn(outputs, targets, cache=None):
            if cache is None:
                cache = {}

            local_ctx = {}

            # Evaluate components with caching
            for name, comp_fn in components.items():
                if name not in cache:
                    cache[name] = comp_fn(outputs, targets, cache)
                local_ctx[name] = cache[name]

            # Add variables/constants
            local_ctx.update(variables)

            # Add torch namespace for functions (e.g., torch.exp)
            local_ctx.update({"torch": torch})

            # Evaluate expression safely
            try:
                loss_val = eval(expression, {"__builtins__": {}}, local_ctx)
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate expression '{expression}': {e}")

            # Apply reduction if needed
            if isinstance(loss_val, torch.Tensor) and loss_val.ndim > 0:
                if reduction == "mean":
                    loss_val = loss_val.mean()
                elif reduction == "sum":
                    loss_val = loss_val.sum()

            return loss_val

        return fn

    def _resolve_params(self, params, outputs, targets, cache=None):
        resolved = {}
        for k, v in params.items():
            if v in ("output", "outputs"):
                resolved[k] = outputs
            elif v in ("target", "targets"):
                resolved[k] = targets
            elif isinstance(v, str) and v.startswith("-") and cache is not None:
                ref = v[1:]
                if ref in cache:
                    resolved[k] = -cache[ref]
                else:
                    raise KeyError(f"Referenced component '{ref}' not found in cache")
            else:
                resolved[k] = v
        return resolved
