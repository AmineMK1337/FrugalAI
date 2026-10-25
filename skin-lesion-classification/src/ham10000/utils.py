import yaml
from dataclasses import dataclass


@dataclass
class Config:
    seed: int
    image_size: int
    train_test_split: float
    model_name: str
    num_classes: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    optimizer: str
    use_class_weights: bool
    use_augmentation: bool
    metadata_csv: str
    data_dir: str
    use_wandb: bool
    
def load_config(path):
    """Load a YAML config file into a typed Config dataclass."""
    with open(path) as f:
        data = yaml.safe_load(f)
        return Config(**data)
