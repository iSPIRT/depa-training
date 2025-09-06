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
import ast

# Unified approved namespace map.
# Keys may be:
#  - root shorthands mapped to live objects: "torch", "nn", "F"
#  - fully-qualified approved prefixes mapped to module import strings
APPROVED_NAMESPACE_MAP = {
    "torch": torch,
    "nn": nn,
    "F": F,
    "torch.nn": "torch.nn",
    "torch.nn.functional": "torch.nn.functional",
    "monai.losses": "monai.losses",
    "kornia.losses": "kornia.losses",
    "piq": "piq",
}

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
      # Resolve object from approved namespaces (supports deep paths like torch.nn.functional.binary_cross_entropy_with_logits)
      obj = _resolve_obj_from_approved_path(cls_path)

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

            # Evaluate expression with a safe arithmetic parser (no calls / attributes)
            try:
                loss_val = _safe_eval_expression(expression, local_ctx)
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
            # Recursive resolution for nested structures
            if isinstance(v, dict):
                resolved[k] = self._resolve_params(v, outputs, targets, cache)
                continue
            if isinstance(v, (list, tuple)):
                seq_type = list if isinstance(v, list) else tuple
                resolved[k] = seq_type(self._resolve_params({"_": x}, outputs, targets, cache)["_"] for x in v)
                continue

            # Direct placeholders
            if v in ("output", "outputs"):
                resolved[k] = outputs
                continue
            if v in ("target", "targets"):
                resolved[k] = targets
                continue

            # References to cached component values
            if isinstance(v, str) and cache is not None:
                if v.startswith("-$"):
                    ref = v[2:]
                    if ref in cache:
                        resolved[k] = -cache[ref]
                        continue
                    raise KeyError(f"Referenced component '{ref}' not found in cache")
                if v.startswith("$"):
                    ref = v[1:]
                    if ref in cache:
                        resolved[k] = cache[ref]
                        continue
                    raise KeyError(f"Referenced component '{ref}' not found in cache")
                if v.startswith("-"):
                    # Backwards-compatible: "-name" refers to negative of cached 'name'
                    ref = v[1:]
                    if ref in cache:
                        resolved[k] = -cache[ref]
                        continue
                    # If not found, fall through to literal string

            # Literal value
            resolved[k] = v
        return resolved


# ----------------------------------------------------------------------
# Security helpers
# ----------------------------------------------------------------------
def _resolve_obj_from_approved_path(path: str):
    """Resolve an attribute object from an approved module path.
    Chooses the longest approved namespace prefix and traverses attributes.
    """
    if not isinstance(path, str):
        raise TypeError("Expected string path")

    # Resolve via longest approved prefix from unified map
    approved_sorted = sorted(APPROVED_NAMESPACE_MAP.keys(), key=len, reverse=True)
    base = None
    for ns in approved_sorted:
        if path == ns or path.startswith(ns + "."):
            base = ns
            break
    if base is None:
        raise ValueError(f"Path '{path}' is not under approved namespaces: {list(APPROVED_NAMESPACE_MAP.keys())}")

    provider = APPROVED_NAMESPACE_MAP[base]
    if isinstance(provider, str):
        # Import the module for string providers
        module = importlib.import_module(provider)
        if path == base:
            raise ValueError(f"Path '{path}' refers to a module, expected a class or function under it")
        remainder = path[len(base) + 1:]
        obj = module
    else:
        # provider is a live module/object (torch, nn, F)
        if path == base:
            raise ValueError(f"Path '{path}' refers to a namespace root, expected a class or function under it")
        remainder = path[len(base) + 1:]
        obj = provider

    for part in remainder.split('.'):
        if part == "":
            raise ValueError(f"Invalid path '{path}'")
        if not hasattr(obj, part):
            raise AttributeError(f"'{obj}' has no attribute '{part}' while resolving '{path}'")
        obj = getattr(obj, part)
    return obj


def _safe_eval_expression(expression: str, names: dict):
    """
    Safely evaluate an arithmetic expression using AST.
    - Supports PEDMAS: +, -, *, /, ** and parentheses
    - Supports unary + and -
    - Disallows function calls, attribute access, subscripting, comprehensions, etc.
    - Names must exist in the provided names dict
    """

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Constants / numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed in expressions")

        # Variables (component outputs or variables)
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise NameError(f"Unknown name in expression: {node.id}")

        # Parentheses are represented implicitly via AST structure

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator in expression")

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            # Optional: uncomment if you want to support floor-div or mod
            # if isinstance(node.op, ast.FloorDiv):
            #     return left // right
            # if isinstance(node.op, ast.Mod):
            #     return left % right
            raise ValueError("Unsupported binary operator in expression")

        # Anything else is forbidden
        forbidden = (
            ast.Call, ast.Attribute, ast.Subscript, ast.Dict, ast.List, ast.Tuple,
            ast.BoolOp, ast.Compare, ast.IfExp, ast.Lambda, ast.ListComp, ast.DictComp,
            ast.GeneratorExp, ast.SetComp, ast.Await, ast.Yield, ast.YieldFrom,
            ast.FormattedValue, ast.JoinedStr
        )
        if isinstance(node, forbidden):
            raise ValueError("Disallowed construct in expression")

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    tree = ast.parse(expression, mode='eval')
    return eval_node(tree)
