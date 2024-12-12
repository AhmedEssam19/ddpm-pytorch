from pydantic import BaseModel
from pathlib import Path


class DatasetConfig(BaseModel):
    name: str


class ModelConfig(BaseModel):
    image_size: int
    timesteps: int
    first_layer_channels: int
    channels_multiplier: list[int]
    num_res_blocks: int
    attn_resolutions: list[int]
    dropout: float


class TrainingConfig(BaseModel):
    batch_size: int
    view_sample_size: int
    num_workers: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    log_interval: int
    results_folder: Path
    accelerator: str
    num_gpus: int
    gradient_clip_val: float


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return cls(**config_dict) 