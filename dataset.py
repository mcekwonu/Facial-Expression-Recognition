import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_class_label


class FERDataset(Dataset):

    def __init__(self, root, transform=None):
        self.path = root
        self.transform = transform
        self.images_path = list()

        for root, dirs, filenames in os.walk(self.path):
            for filename in filenames:
                self.images_path.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        image = cv2.imread(img_path)
        annotations = get_class_label(self.path)
        class_label = img_path.split("/")[-2]
        label = annotations[class_label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)



