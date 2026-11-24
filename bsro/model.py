from __future__ import annotations

from pathlib import Path

import torch
import yaml
from bs_roformer import BSRoformer

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
    for key in _IGNORED_MODEL_KEYS:
        params.pop(key, None)
    return BSRoformer(**params)


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
