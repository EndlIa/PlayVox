"""Config loading for the PlayVox stage-1 pipeline.

The released checkpoints ship their own ``pipeline.yaml`` / ``ss_generator.yaml``
/ ``ss_decoder.yaml``. They are expected to reference this package
(``forehoi.*``); the assets live outside this repository and are prepared
accordingly (see ``setup/ASSETS.md``).
"""
from __future__ import annotations

import os

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def build_from_config(path: str, **overrides):
    config = load_config(path)
    config.workspace_dir = os.path.dirname(path)
    for key, value in overrides.items():
        config[key] = value
    return instantiate(config)
