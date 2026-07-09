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


from typing import Any, Dict, List, Tuple, Callable
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


APPROVED_NAMESPACES = {
    "torch": torch,
    "nn": nn,
    "F": F,
}

# Security controls
ALLOWED_OP_PREFIXES = {"F.", "torch.nn.functional."}  # Allow only torch.nn.functional.* by default
ALLOWED_OPS = {
    "torch.cat", "torch.stack", "torch.concat", "torch.flatten", "torch.reshape", "torch.permute", "torch.transpose", 
    "torch.unsqueeze", "torch.squeeze", "torch.chunk", "torch.split", "torch.gather", "torch.index_select", "torch.narrow",
    "torch.sum", "torch.mean", "torch.std", "torch.var", "torch.max", "torch.min", "torch.argmax", "torch.argmin", "torch.norm",
    "torch.exp", "torch.log", "torch.log1p", "torch.sigmoid", "torch.tanh", "torch.softmax", "torch.log_softmax", "torch.relu", "torch.gelu",
    "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm", "torch.einsum",
    "torch.roll", "torch.flip", "torch.rot90", "torch.rot180", "torch.rot270", "torch.rot360",
}

# Denylist of potentially dangerous kwarg names (case-insensitive)
DENYLIST_ARG_NAMES = {
    "out",  # in-place writes to user-provided buffers
    "file", "filename", "path", "dir", "directory",  # filesystem
    "map_location",  # avoid device remap surprises
}

# DoS safeguards
MAX_FORWARD_STEPS = 200
MAX_OPS_PER_STEP = 10


def _resolve_submodule(path: str) -> Any:
    """Resolve dotted path like 'nn.Conv2d' or 'torch.sigmoid' to an object.
    Raises AttributeError if resolution fails.
    """
    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        parts = path.split(".")
        if parts[0] in APPROVED_NAMESPACES:
            obj = APPROVED_NAMESPACES[parts[0]]
        else:
            # allow direct module names like 'math' if needed
            raise AttributeError(f"Unknown root namespace '{parts[0]}' in path '{path}'")
        for p in parts[1:]:
            try:
                obj = getattr(obj, p)
            except AttributeError:
                raise AttributeError(f"Could not resolve attribute '{p}' in path '{path}'")
        return obj
    except Exception as e:
        raise RuntimeError(f"Error resolving dotted path '{path}': {str(e)}") from e


def _replace_placeholders(obj: Any, params: Dict[str, Any]) -> Any:
    """Recursively replace strings of the form '$name' using params mapping."""
    try:
        if isinstance(obj, str) and obj.startswith("$"):
            key = obj[1:]
            if key not in params:
                raise KeyError(f"Placeholder '{obj}' not found in params {params}")
            return params[key]
        elif isinstance(obj, dict):
            try:
                return {k: _replace_placeholders(v, params) for k, v in obj.items()}
            except Exception as e:
                raise RuntimeError(f"Error replacing placeholders in dict: {str(e)}") from e
        elif isinstance(obj, (list, tuple)):
            try:
                seq_type = list if isinstance(obj, list) else tuple
                return seq_type(_replace_placeholders(x, params) for x in obj)
            except Exception as e:
                raise RuntimeError(f"Error replacing placeholders in sequence: {str(e)}") from e
        else:
            return obj
    except Exception as e:
        raise RuntimeError(f"Error in placeholder replacement: {str(e)}") from e


class ModelFactory:
    """Factory for building PyTorch nn.Module instances from config dicts.

    Public API:
        ModelFactory.load_from_dict(config: dict) -> nn.Module
    """

    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> nn.Module:
        """Create an nn.Module instance from a top-level config.

        The config may define 'submodules' (a dict of reusable component templates) and
        a top-level 'layers' and 'forward' graph. Submodules are used by layers that have
        a 'submodule' key and are instantiated with their provided params.
        """
        try:
            if not isinstance(config, dict):
                raise TypeError("Config must be a dictionary")
                
            submodules_defs = config.get("submodules", {})

            def create_instance_from_def(def_cfg: Dict[str, Any], provided_params: Dict[str, Any]):
                try:
                    # Replace placeholders in the def_cfg copy
                    # Deep copy not strictly necessary since we replace on the fly
                    replaced_cfg = {
                        k: (_replace_placeholders(v, provided_params) if k in ("layers",) or isinstance(v, dict) else v)
                        for k, v in def_cfg.items()
                    }
                    # Build module from replaced config (submodule templates should not themselves contain further 'submodules')
                    return cls._build_module_from_config(replaced_cfg, submodules_defs)
                except Exception as e:
                    raise RuntimeError(f"Error creating instance from definition: {str(e)}") from e

            # When a layer entry references a 'submodule', we instantiate it using template from submodules_defs
            return cls._build_module_from_config(config, submodules_defs)
        except Exception as e:
            raise RuntimeError(f"Error loading model from config: {str(e)}") from e

    @classmethod
    def _build_module_from_config(cls, config: Dict[str, Any], submodules_defs: Dict[str, Any]) -> nn.Module:
        try:
            layers_cfg = config.get("layers", {})
            forward_cfg = config.get("forward", [])
            input_names = config.get("input", [])
            output_names = config.get("output", [])

            # Create dynamic module class
            class DynamicModule(nn.Module):
                def __init__(self):
                    try:
                        super().__init__()
                        # ModuleDict to register submodules / layers
                        self._layers = nn.ModuleDict()
                        # Save forward graph and io names
                        self._forward_cfg = forward_cfg
                        self._input_names = input_names
                        self._output_names = output_names

                        # Build each layer / submodule
                        for name, entry in layers_cfg.items():
                            try:
                                if "class" in entry:
                                    cls_obj = _resolve_submodule(entry["class"])  # e.g. nn.Conv2d
                                    if not (isinstance(cls_obj, type) and issubclass(cls_obj, nn.Module)):
                                        raise TypeError(f"Layer '{name}' class must be an nn.Module subclass, got {cls_obj}")
                                    params = entry.get("params", {})
                                    inst_params = _replace_placeholders(params, {})  # top-level layers likely have no placeholders
                                    module = cls_obj(**inst_params)
                                    self._layers[name] = module
                                elif "submodule" in entry:
                                    sub_name = entry["submodule"]
                                    if sub_name not in submodules_defs:
                                        raise KeyError(f"Submodule '{sub_name}' not found in submodules definitions")
                                    sub_def = submodules_defs[sub_name]
                                    provided_params = entry.get("params", {})
                                    # Replace placeholders inside sub_def using provided_params
                                    # We create a fresh instance of submodule by calling helper
                                    sub_inst = cls._instantiate_submodule(sub_def, provided_params, submodules_defs)
                                    self._layers[name] = sub_inst
                                else:
                                    raise KeyError(f"Layer '{name}' must contain either 'class' or 'submodule' key")
                            except Exception as e:
                                raise RuntimeError(f"Error building layer '{name}': {str(e)}") from e
                    except Exception as e:
                        raise RuntimeError(f"Error initializing DynamicModule: {str(e)}") from e

                def forward(self, *args, **kwargs):
                    try:
                        # Map inputs
                        env: Dict[str, Any] = {}
                        # assign by position
                        for i, in_name in enumerate(self._input_names):
                            if i < len(args):
                                env[in_name] = args[i]
                            elif in_name in kwargs:
                                env[in_name] = kwargs[in_name]
                            else:
                                raise ValueError(f"Missing input '{in_name}' for forward; provided args={len(args)}, kwargs keys={list(kwargs.keys())}")

                        # Execute forward graph
                        if len(self._forward_cfg) > MAX_FORWARD_STEPS:
                            raise RuntimeError(f"Too many forward steps: {len(self._forward_cfg)} > {MAX_FORWARD_STEPS}. This is a security feature to prevent infinite loops.")

                        for idx, step in enumerate(self._forward_cfg):
                            try:
                                ops = step.get("ops", [])
                                if isinstance(ops, (list, tuple)) and len(ops) > MAX_OPS_PER_STEP:
                                    raise RuntimeError(f"Too many ops in step {idx}: {len(ops)} > {MAX_OPS_PER_STEP}")
                                inputs_spec = step.get("input", [])
                                out_name = step.get("output", None)

                                # Resolve input tensors for this step
                                # inputs_spec might be: ['x'] or ['x1','x2'] or [['x3','encoded_feature']]
                                if len(inputs_spec) == 1 and isinstance(inputs_spec[0], (list, tuple)):
                                    args_list = [env[n] for n in inputs_spec[0]]
                                else:
                                    args_list = [env[n] for n in inputs_spec]

                                # Apply ops sequentially
                                current = args_list
                                for op in ops:
                                    try:
                                        # op can be string like 'conv1' or dotted 'F.relu'
                                        # or can be a list like ['torch.flatten', {'start_dim':1}]
                                        op_callable, op_kwargs = self._resolve_op(op)
                                        # Validate kwargs denylist
                                        for k in op_kwargs.keys():
                                            if isinstance(k, str) and k.lower() in DENYLIST_ARG_NAMES:
                                                raise PermissionError(f"Denied kwarg '{k}' for op '{op}'")

                                        # If op_callable is a module in self._layers, call with module semantics
                                        if isinstance(op_callable, str) and op_callable in self._layers:
                                            module = self._layers[op_callable]
                                            # if current is list of multiple args, pass them all
                                            if isinstance(current, (list, tuple)) and len(current) > 1:
                                                result = module(*current)
                                            else:
                                                result = module(current[0])
                                        else:
                                            # op_callable is a real callable object

                                            if op_callable in {torch.cat, torch.stack}: # Ops that require a sequence input (instead of varargs)
                                                # Wrap current into a list
                                                result = op_callable(list(current), **op_kwargs)
                                            elif isinstance(current, (list, tuple)):
                                                result = op_callable(*current, **op_kwargs)
                                            else:
                                                result = op_callable(current, **op_kwargs)

                                        # prepare current for next op
                                        current = [result]
                                    except Exception as e:
                                        raise RuntimeError(f"Error applying operation '{op}': {str(e)}") from e

                                # write outputs back into env
                                if out_name is None:
                                    continue
                                if isinstance(out_name, (list, tuple)):
                                    # if step produces multiple outputs (rare), try unpacking
                                    if len(out_name) == 1:
                                        env[out_name[0]] = current[0]
                                    else:
                                        # try to unpack
                                        try:
                                            for k, v in zip(out_name, current[0]):
                                                env[k] = v
                                        except Exception as e:
                                            raise RuntimeError(f"Could not assign multiple outputs for step {step}: {e}")
                                else:
                                    env[out_name] = current[0]
                            except Exception as e:
                                raise RuntimeError(f"Error executing forward step: {str(e)}") from e

                        # Build function return
                        if len(self._output_names) == 0:
                            return None
                        if len(self._output_names) == 1:
                            return env[self._output_names[0]]
                        return tuple(env[n] for n in self._output_names)
                    except Exception as e:
                        raise RuntimeError(f"Error in forward pass: {str(e)}") from e

                def _resolve_op(self, op_spec):
                    """Return (callable_or_module_name, kwargs)

                    If op_spec is a string and matches a layer name -> returns (layer_name_str, {}).
                    If op_spec is a string dotted path -> resolve dotted and return (callable, {}).
                    If op_spec is a list like ["torch.flatten", {"start_dim":1}] -> resolve and return (callable, kwargs)
                    """
                    try:
                        # module reference by name
                        if isinstance(op_spec, str):
                            if op_spec in self._layers:
                                return (op_spec, {})
                            # dotted function (F.relu, torch.sigmoid)
                            if not _is_allowed_op_path(op_spec):
                                raise PermissionError(f"Operation '{op_spec}' is not allowed")
                            callable_obj = _resolve_submodule(op_spec)
                            if not callable(callable_obj):
                                raise TypeError(f"Resolved object for '{op_spec}' is not callable")
                            return (callable_obj, {})
                        elif isinstance(op_spec, (list, tuple)):
                            if len(op_spec) == 0:
                                raise ValueError("Empty op_spec list")
                            path = op_spec[0]
                            kwargs = op_spec[1] if len(op_spec) > 1 else {}
                            if not _is_allowed_op_path(path):
                                raise PermissionError(f"Operation '{path}' is not allowed")
                            callable_obj = _resolve_submodule(path)
                            if not callable(callable_obj):
                                raise TypeError(f"Resolved object for '{path}' is not callable")
                            return (callable_obj, kwargs)
                        else:
                            raise TypeError(f"Unsupported op spec type: {type(op_spec)}")
                    except Exception as e:
                        raise RuntimeError(f"Error resolving operation '{op_spec}': {str(e)}") from e

            # Instantiate dynamic module and return
            dyn = DynamicModule()
            return dyn
        except Exception as e:
            raise RuntimeError(f"Error building module from config: {str(e)}") from e

    @classmethod
    def _instantiate_submodule(cls, sub_def: Dict[str, Any], provided_params: Dict[str, Any], submodules_defs: Dict[str, Any]) -> nn.Module:
        """Instantiate a submodule defined in 'submodules' using provided_params to replace placeholders.

        provided_params are used to replace occurrences of strings like '$in_ch' inside the sub_def's 'layers' params.
        """
        try:
            # Deep replace placeholders within sub_def copy
            # We'll construct a new config where the "layers"->"params" are substituted
            replaced = {}
            for k, v in sub_def.items():
                try:
                    if k == "layers":
                        new_layers = {}
                        for lname, lentry in v.items():
                            new_entry = dict(lentry)
                            if "params" in lentry:
                                new_entry["params"] = _replace_placeholders(lentry["params"], provided_params)
                            new_layers[lname] = new_entry
                        replaced[k] = new_layers
                    else:
                        # copy other keys directly (input/forward/output)
                        replaced[k] = v
                except Exception as e:
                    raise RuntimeError(f"Error processing key '{k}': {str(e)}") from e

            # Now build a module from this replaced config. This call may in turn instantiate nested submodules.
            return cls._build_module_from_config(replaced, submodules_defs)
        except Exception as e:
            raise RuntimeError(f"Error instantiating submodule: {str(e)}") from e


def _is_allowed_op_path(path: str) -> bool:
    if any(path.startswith(p) for p in ALLOWED_OP_PREFIXES):
        return True
    return path in ALLOWED_OPS