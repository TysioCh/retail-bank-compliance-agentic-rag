from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuantizationConfig:
    model_path: Path | None = None
    calibration_data: Path | None = None
    output_dir: Path | None = None
    bits: int = 4

    def to_command(self) -> list[str]:
        return [
            "autoawq",
            "--model", str(self.model_path),
            "--calib_data", str(self.calibration_data),
            "--w_bit", str(self.bits),
            "--output_dir", str(self.output_dir),
        ]


def quantize_model(config: QuantizationConfig) -> None:
    print(" ".join(config.to_command()))
