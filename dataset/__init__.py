# dataset/__init__.py
from .imagenet_loader import get_imagenet_dataloader

def get_dataloaders(config, dataset_name):
    if dataset_name == 'imagenet':
        return get_imagenet_dataloader(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")