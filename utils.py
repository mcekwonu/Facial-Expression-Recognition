import os
import numpy as np
import torch
import splitfolders

from torch.utils.data import DataLoader
from torchvision import transforms


def create_val_test(source_dir, target_dir, split_ratio=(0.9, 0.1), seed=142):
    os.makedirs(target_dir, exist_ok=True)

    assert (
            len(split_ratio) == 2 or len(split_ratio) == 3
    ), f"{split_ratio} must be a contain two or three values!"

    if len(split_ratio) == 2:
        return splitfolders.ratio(source_dir, target_dir, seed=seed, ratio=split_ratio)

    elif len(split_ratio) == 3:
        return splitfolders.ratio(source_dir, target_dir, seed=seed, ratio=split_ratio)


class Labels:
    def __init__(self, path):
        self.classes = {}

        for root, dirs, filenames in os.walk(path):
            for idx, name in enumerate(dirs):
                if name not in self.classes:
                    self.classes[idx] = name

    def _print(self):
        print(self.classes)

    def __getitem__(self, item):
        return self.classes[item]


def get_class_label(path, return_idx=True):
    classes = {}

    for root, dirs, filenames in os.walk(path):
        for idx, name in enumerate(dirs):
            if name not in classes:
                classes[idx] = name

    if return_idx:
        return {value: key for key, value in classes.items()}


def get_loader(dataset, batch_size=4, num_workers=2, shuffle=False, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, pin_memory=pin_memory)


def composed_transform(mean, std, argumentation=None):
    if argumentation:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def compute_mean_and_std(loader):
    images, _ = next(iter(loader))
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    if images.size(1) == 1:
        return mean[0], std[0]
    else:
        return mean, std


def convert_to_time(time):
    """Print the training time in days: hours: minutes: seconds format."""
    days = time // (24 * 60 * 60)
    time %= (24 * 60 * 60)
    hrs = time // (60 * 60)
    time %= (60 * 60)
    mins = time // 60
    time %= 60
    secs = time
    msg = f"Training completed in {days:.0f} days: {hrs:.0f} hours: {mins:.0f} minutes: {secs:.0f} seconds"
    print(msg)
    return msg


def timer_log(filepath, msg):
    with open(filepath, "a") as f:
        f.write(msg)
