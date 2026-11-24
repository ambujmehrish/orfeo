from __future__ import annotations

from pathlib import Path

import torch
import yaml
from bs_roformer import BSRoformer
from bs_roformer.bs_roformer import MaskEstimator
from torch.nn import ModuleList

# Keys present in the MSST training configs that lucidrains' BSRoformer does not accept.
_IGNORED_MODEL_KEYS = ("linear_transformer_depth", "use_torch_checkpoint", "skip_connection")


class _ConfigLoader(yaml.SafeLoader):
    pass


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_ConfigLoader)


def build_model(config: dict) -> BSRoformer:
    params = dict(config["model"])
    mlp_expansion_factor = params.pop("mlp_expansion_factor", 4)
    for key in _IGNORED_MODEL_KEYS:
        params.pop(key, None)
    model = BSRoformer(**params)
    # MSST checkpoints may use a narrower mask-estimator MLP than lucidrains' fixed default of 4.
    if mlp_expansion_factor != 4:
        dim_inputs = model.mask_estimators[0].dim_inputs
        model.mask_estimators = ModuleList(
            MaskEstimator(
                params["dim"],
                dim_inputs,
                params["mask_estimator_depth"],
                mlp_expansion_factor=mlp_expansion_factor,
            )
            for _ in range(params["num_stems"])
        )
    return model


def load_checkpoint(model: BSRoformer, path: str | Path) -> BSRoformer:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def stem_names(config: dict) -> list[str]:
    training = config["training"]
    target = training.get("target_instrument")
    if target is None:
        return list(training["instruments"])
    return [target]


def residual_name(config: dict) -> str | None:
    training = config["training"]
    target = training.get("target_instrument")
    if target is None:
        return None
    others = [name for name in training["instruments"] if name != target]
    return others[0] if len(others) == 1 else "residual"
