from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RerankerTrainingConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    train_data: Path | None = None
    eval_data: Path | None = None
    output_dir: Path | None = None
    learning_rate: float = 2e-5
    num_epochs: int = 1
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2

    def to_command(self) -> list[str]:
        return [
            "swift", "sft",
            "--model", self.model_name,
            "--dataset", str(self.train_data),
            "--val_dataset", str(self.eval_data),
            "--output_dir", str(self.output_dir),
            "--learning_rate", str(self.learning_rate),
            "--num_train_epochs", str(self.num_epochs),
            "--per_device_train_batch_size", str(self.per_device_batch_size),
            "--gradient_accumulation_steps", str(self.gradient_accumulation_steps),
            "--loss_type", "generative_reranker",
        ]


def train_reranker(config: RerankerTrainingConfig) -> None:
    """Generate the training command for the real runner."""
    print(" ".join(config.to_command()))
