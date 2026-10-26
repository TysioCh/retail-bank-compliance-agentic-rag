from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMSftConfig:
    model_name: str = "Qwen/Qwen3-8B"
    train_data: Path | None = None
    eval_data: Path | None = None
    output_dir: Path | None = None
    learning_rate: float = 2e-5
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True

    def to_command(self) -> list[str]:
        command = [
            "llamafactory-cli", "train",
            "--model_name_or_path", self.model_name,
            "--dataset", str(self.train_data),
            "--output_dir", str(self.output_dir),
            "--learning_rate", str(self.learning_rate),
            "--num_train_epochs", str(self.num_epochs),
            "--per_device_train_batch_size", str(self.per_device_batch_size),
            "--gradient_accumulation_steps", str(self.gradient_accumulation_steps),
        ]
        if self.use_lora:
            command.append("--finetuning_type")
            command.append("lora")
        return command


def train_llm_sft(config: LLMSftConfig) -> None:
    print(" ".join(config.to_command()))
