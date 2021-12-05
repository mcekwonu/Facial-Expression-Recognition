import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FERDataset

from utils import *
from network import *
from lr_finder import *
from trainer import Trainer

TRAIN_DIR = "data/train"
SAVE_DIR = "logs/plots"
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda"

train_ds = FERDataset(root=TRAIN_DIR, transform=transforms.ToTensor())
t_loader = get_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                      pin_memory=PIN_MEMORY)
mean, std = compute_mean_and_std(t_loader)

# Reload and prepare dataset with appropriate transformation for training
train_transform = composed_transform(mean=mean, std=std, argumentation=True)
train_dataset = FERDataset(root=TRAIN_DIR, transform=train_transform)
train_loader = get_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY)

MODEL = FerNet54().to(DEVICE)
LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=1e-1)

lr_find = LearningRateFinder(MODEL, LOSS_FN, OPTIMIZER, DEVICE)
lr_find.fit(train_loader, steps=1200)

# Save and display the plot of the result
lr_find.plot(save_dir=SAVE_DIR, verbose=True)
lr_find.reset
