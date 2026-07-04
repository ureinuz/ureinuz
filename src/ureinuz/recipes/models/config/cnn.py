
from dataclasses import dataclass



@dataclass(frozen=True)
class CNNModelConfig:
    num_classes: int
    depths: list[int]
    dims: list[int]
    drop_path_rate: float = 0.0