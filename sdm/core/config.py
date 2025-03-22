import yaml

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass
class DataConfig:
    data_dir: str
    input_path: str
    hf_dataset: bool
    dataset_name: str


@dataclass
class ModelConfig:
    n: int
    p: int
    encoder_name: str


@dataclass
class TrainConfig:
    epochs: int
    max_tokens: int
    train: bool


class Config:
    def __init__(self, data=None, model=None, train=None):
        # Convert each section to a SimpleNamespace for dot access
        self.data = self._dict_to_namespace(data) if data else SimpleNamespace()
        self.model = self._dict_to_namespace(model) if model else SimpleNamespace()
        self.train = self._dict_to_namespace(train) if train else SimpleNamespace()

    @classmethod
    def from_yaml(cls, yaml_path="./config/config.yml"):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Ensure all necessary sections exist
        for section in ["data", "model", "train"]:
            if section not in config_dict:
                config_dict[section] = {}

        return cls(**config_dict)

    def _dict_to_namespace(self, d):
        """Convert a dictionary to a SimpleNamespace for dot access"""
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, self._dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

    def _namespace_to_dict(self, namespace):
        """Convert a SimpleNamespace object back to a dictionary"""
        result = {}
        for key, value in namespace.__dict__.items():
            if isinstance(value, SimpleNamespace):
                result[key] = self._namespace_to_dict(value)
            else:
                result[key] = value
        return result

    def save(self, path):
        # Convert SimpleNamespace objects back to dictionaries
        config_dict = {
            "data": self._namespace_to_dict(self.data),
            "model": self._namespace_to_dict(self.model),
            "train": self._namespace_to_dict(self.train),
        }

        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
