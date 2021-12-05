import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import FERDataset
from utils import *
from network import *


def predict_image(model_path, model, dataloader, device="cpu"):
    device = torch.device(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        for img, y in dataloader:
            y_pred = model(img).to(device)
            y_pred = F.log_softmax(y_pred, dim=1)
            pred = y_pred.argmax(dim=1)

    return img, y, pred


def main():
    test_dir = "data/test"
    batch_size = 1
    num_workers = 2

    test_ds = FERDataset(root=test_dir, transform=transforms.ToTensor())
    t_loader = get_loader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_mean, test_std = compute_mean_and_std(t_loader)

    test_transform = composed_transform(mean=test_mean, std=test_std, argumentation=False)
    test_dataset = FERDataset(root=test_dir, transform=test_transform)
    test_loader = get_loader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = FerNet54()
    label = Labels(path=test_dir)
    model_path = "logs/saved_models/ferresnet.pth"
    start = time.time()
    image, target, predicted = predict_image(model_path, model, test_loader)
    elapsed = time.time() - start
    image = image.squeeze(0).numpy().transpose(1, 2, 0)
    pred_label = label[predicted.item()]
    target_label = label[target.item()]
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.title(f"Target: {target_label}   Predicted: {pred_label}")
    plt.xticks([]), plt.yticks([])
    plt.savefig(f"logs/plots/predicted_{pred_label}.png", dpi=600, bbox_inches="tight")
    plt.show()
    print(f"Completed in: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
