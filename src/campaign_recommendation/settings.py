from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    raw: dict
    root_dir: Path

    @property
    def training_data_path(self) -> Path:
        return self.root_dir / self.raw["training_data_path"]

    @property
    def model_output_dir(self) -> Path:
        return self.root_dir / self.raw["model_output_dir"]

    @property
    def recommendation_output_path(self) -> Path:
        return self.root_dir / self.raw["recommendation_output_path"]


def load_config(config_path: str | Path | None = None) -> AppConfig:
    root = Path(__file__).resolve().parents[2]
    resolved = root / "config" / "default_config.json" if config_path is None else Path(config_path)
    with resolved.open("r", encoding="utf-8") as handle:
        return AppConfig(raw=json.load(handle), root_dir=root)
