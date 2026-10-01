"""Shared YAML configuration helpers for command-line tasks."""

from pathlib import Path

import yaml


def load_path_config(config_path):
    """Load paths, resolving relative data paths from the repository root."""
    if not config_path:
        config_path = "configs/paths.yaml"
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    project_root = path.resolve().parent.parent

    def resolve(value):
        if not value:
            return value
        candidate = Path(str(value)).expanduser()
        return str(candidate if candidate.is_absolute() else (project_root / candidate).resolve())

    for key in ("project_root", "checkpoint"):
        if key in config:
            config[key] = resolve(config[key])

    spatial = config.get("spatial_gene_expression", {}) or {}
    roots = spatial.get("data_roots", {}) or {}
    for key, value in roots.items():
        roots[key] = resolve(value)
    if spatial.get("data_root"):
        spatial["data_root"] = resolve(spatial["data_root"])

    for section, keys in {
        "tumor_detection": ("root_dir", "feature_cache_dir"),
        "hemit": ("root", "labels_csv", "feature_cache_dir"),
        "visium_hd": (
            "base_dir", "train_base_dir", "test_base_dir", "he_tif",
            "train_he_tif", "test_he_tif", "hvg_file", "feature_dir",
            "train_feature_path", "test_feature_path",
        ),
    }.items():
        values = config.get(section, {}) or {}
        for key in keys:
            if values.get(key):
                values[key] = resolve(values[key])
    return config
