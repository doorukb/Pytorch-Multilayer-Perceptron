from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from torchmlp.model import ActivationName
OptimizerName = Literal["sgd", "adam"]

# the configuration for the training of the model
@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    hidden_sizes: list[int] = field(default_factory=lambda: [5])
    input_dim: int = 2
    output_dim: int = 1
    dropout: float = 0.0
    activation: ActivationName = "sigmoid"
    optimizer: OptimizerName = "sgd"
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42
    n_samples: int = 1000
    device: str | None = None

    @property
    def layer_sizes(self) -> list[int]:
        return [self.input_dim, *self.hidden_sizes, self.output_dim]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")