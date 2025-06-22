"""
This module defines:
    1. dataset loader
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Optional, Callable, Tuple, Dict, List
import os
from PIL import Image

class CustomImageDataset(Dataset):
    """Custom dataset that respects user-defined class_to_idx mapping"""
    
    def __init__(self, root_dir: str, class_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = list(class_to_idx.keys())
        
        # Build file list
        self.samples = []
        self._build_samples()
    
    def _build_samples(self):
        """Build list of (image_path, class_idx) tuples"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DatasetLoader:
    """
    Dataset loader for ImageNet-structured dataset
    """

    def __init__(self,
                 dataset_path: str,
                 label_set: List[str],
                 transform: Optional[Callable] = None,
                 batch_size: int = 32,
                 num_workers: int = 4):
        
        self.dataset_path = dataset_path
        self.label_set = label_set
        self.transform = transform or self._get_default_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create class_to_idx mapping
        self.class_to_idx = {label: idx for idx, label in enumerate(label_set)}
        self.classes = label_set

        self.train_path = os.path.join(dataset_path, 'train')
        self.test_path = os.path.join(dataset_path, 'test')

    def _get_default_transform(self):
        """Dafault transform, using statistics from ImageNet."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_train_loader(self, train_transform) -> DataLoader:
        """Get training data loader"""
        train_set = CustomImageDataset(
            root_dir=self.train_path,
            class_to_idx=self.class_to_idx,
            transform=train_transform
        )
        
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_loader(self, test_transform) -> DataLoader:
        """Get testing data loader"""
        test_set = CustomImageDataset(
            root_dir=self.test_path,
            class_to_idx=self.class_to_idx,
            transform=test_transform
        )
        
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_names(self) -> list:
        """Get class names from the dataset"""
        return self.classes

    def get_class_to_idx(self) -> dict:
        """Get class to index mapping"""
        return self.class_to_idx

    def get_dataset_info(self) -> dict:
        """Get dataset information"""
        train_set = CustomImageDataset(
            root_dir=self.train_path,
            class_to_idx=self.class_to_idx,
        )
        test_set = CustomImageDataset(
            root_dir=self.test_path,
            class_to_idx=self.class_to_idx,
        )
        
        return {
            'num_classes': len(self.classes),
            'class_names': self.classes,
            'class_to_idx': self.class_to_idx,
            'train_size': len(train_set),
            'test_size': len(test_set)
        }


if __name__ == "__main__":
    # 038_HAM10000 dataset example
    dataset_name = "038_HAM10000"
    from datasetProcesser import DATASET_META_CLS, LOCAL_DATASET_BASE
    import json

    with open(DATASET_META_CLS, 'r') as f:
        dataset_meta = json.load(f)
    label_set = dataset_meta["D"+dataset_name]['label_set']

    dataset_path = os.path.join(LOCAL_DATASET_BASE, dataset_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loader = DatasetLoader(dataset_path, label_set, transform, batch_size=32, num_workers=4)

    # Get train and test loaders
    train_loader = loader.get_train_loader(transform)
    test_loader = loader.get_test_loader(transform)

    # Print dataset information
    dataset_info = loader.get_dataset_info()
    print(f"Dataset Info: {dataset_info}")