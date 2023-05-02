from .LandDataset import LandDataset
from .ShipDataset import ShipDataset


def build_dataset(root_dir:str, class_labels:list, transforms=None):
    dataset = LandDataset(root_dir, class_labels, transforms)
    return dataset


def build_dataset_ships(imgs_dir:str, meta_path:str, transforms=None):
    dataset = ShipDataset(imgs_dir, meta_path, transforms)
    return dataset