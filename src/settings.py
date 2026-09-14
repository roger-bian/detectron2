"""Load per-run settings for train.py and inference.py from settings.yaml.

Validation is strict: a section must contain exactly the expected keys, so a
typo ("score_threshold") is an error rather than a silently ignored no-op.
"""

import os
import sys
from types import SimpleNamespace

import yaml


DEFAULT_PATH = "settings.yaml"

SCHEMA = {
    "train": [
        "dataset_name", "version", "num_classes", "batch_size", "epochs",
        "eval_period", "checkpoint_period", "config_file", "config", "weights",
    ],
    "inference": [
        "config", "weights", "path", "output_dir", "score_thresh",
        "nms_thresh", "min_size_test", "max_size_test", "class_names",
        "color", "mask_alpha", "box_color", "no_display", "cpu",
    ],
}

# Keys that must not be null, per section.
REQUIRED = {
    "train": ["dataset_name", "version", "num_classes", "batch_size", "epochs",
              "eval_period", "checkpoint_period", "config_file"],
    "inference": ["config", "path", "output_dir", "score_thresh", "nms_thresh",
                  "mask_alpha", "no_display", "cpu"],
}


class SettingsError(Exception):
    """Raised when settings.yaml is missing, malformed, or inconsistent."""


def _check_rgb(name, value):
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SettingsError(f"'{name}' must be a list of 3 values [R, G, B], got: {value!r}")
    for c in value:
        if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 255:
            raise SettingsError(f"'{name}' values must be integers 0-255, got: {value!r}")


def _validate(section, cfg):
    if section == "inference":
        _check_rgb("color", cfg["color"])
        _check_rgb("box_color", cfg["box_color"])
        if cfg["class_names"] is not None and not isinstance(cfg["class_names"], list):
            raise SettingsError(
                f"'class_names' must be a list, e.g. [mask, screw, hook], "
                f"got: {cfg['class_names']!r}"
            )


def load_settings(section, path=None):
    """Return the given section of settings.yaml as a SimpleNamespace.

    `path` defaults to the first command-line argument, else settings.yaml.
    """
    if path is None:
        path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    if not os.path.isfile(path):
        raise SettingsError(f"Settings file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or section not in data:
        raise SettingsError(f"{path} has no '{section}:' section")

    cfg = data[section]
    if not isinstance(cfg, dict):
        raise SettingsError(f"'{section}:' in {path} must be a mapping of settings")

    expected = set(SCHEMA[section])
    got = set(cfg)
    if unknown := sorted(got - expected):
        raise SettingsError(
            f"Unknown key(s) in '{section}' section of {path}: {', '.join(unknown)}"
        )
    if missing := sorted(expected - got):
        raise SettingsError(
            f"Missing key(s) in '{section}' section of {path}: {', '.join(missing)}"
        )
    if empty := sorted(k for k in REQUIRED[section] if cfg[k] is None):
        raise SettingsError(
            f"Key(s) in '{section}' section of {path} must not be null: {', '.join(empty)}"
        )

    _validate(section, cfg)
    return SimpleNamespace(**cfg)
