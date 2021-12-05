import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import FERDataset
from utils import *
from network import *


class Trainer:

    def __init__(self, model,
                 optimizer,
                 loss_fn,
                 batch_size=8,
                 epochs=100,
                 learning_rate=1e-4,
                 train_dir="data/train",
                 val_dir="data/val",
                 device="cuda",
                 best_loss=float("inf"),
                 num_workers=4,
                 log_interval=100,
                 scheduler=None,
                 pin_memory=True,
                 verbose=False,
                 logs_dir="logs"
                 ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_epochs = epochs
        self.lr = learning_rate
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.verbose = verbose
        self.scheduler = scheduler
        self.best_loss = best_loss
        self.losses = {"train": [], "valid": []}
        self.accuracy = {"train": [], "valid": []}

        TRAIN_DIR = train_dir
        VAL_DIR = val_dir

        self.create_dir(os.path.join(logs_dir, "checkpoints"))
        self.create_dir(os.path.join(logs_dir, "history"))
        self.create_dir(os.path.join(logs_dir, "saved_models"))
        self.create_dir(os.path.join(logs_dir, "plots"))

        train_ds = FERDataset(root=train_dir, transform=transforms.ToTensor())
        val_ds = FERDataset(root=val_dir, transform=transforms.ToTensor())

        train_loader = get_loader(train_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin_memory)
        val_loader = get_loader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin_memory)
        t_mean, t_std = compute_mean_and_std(train_loader)
        val_mean, val_std = compute_mean_and_std(val_loader)

        train_transform = composed_transform(mean=t_mean, std=t_std, argumentation=True)
        val_transform = composed_transform(mean=val_mean, std=val_std, argumentation=True)

        train_dataset = FERDataset(root=TRAIN_DIR, transform=train_transform)
        val_dataset = FERDataset(root=VAL_DIR, transform=val_transform)

        self.train_loader = get_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_memory)
        self.val_loader = get_loader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_memory)

        if scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5)

    def fit(self):
        start = time.time()
        for self.epoch in range(self.num_epochs):
            print("\nEpoch {:3d}/{}: \tPhase: Training".format(self.epoch + 1, self.num_epochs))
            print("-" * 32)
            self.on_epoch_train()
            self.on_epoch_valid()
            torch.cuda.empty_cache()
        time_elapsed = time.time() - start

        np.savez(f"logs/history/losses", **self.losses)
        np.savez(f"logs/history/acc", **self.accuracy)
        msg = convert_to_time(time_elapsed)
        timer_log(f"logs/history/training_time.txt", msg)

    def on_epoch_train(self):
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = f"logs/checkpoints/ckpt_{date}.pth"

        state = {
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        self.model.train()
        train_loss = []
        train_acc = []
        total_acc = 0.
        total_count = 0

        for step, (image, targets) in enumerate(self.train_loader):
            image, targets = image.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.loss_fn(pred, targets)
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            total_acc += (pred == targets.view_as(pred)).sum().item()
            total_count += self.train_loader.batch_size
            acc = 100 * total_acc / total_count
            train_acc.append(acc)

            if step % self.log_interval == 0:
                print(
                    "Current step: {:3d}\t  Loss: {:.5f}\t  Accuracy: {:.2f}%".format(
                        step, loss.item(), acc)
                )
                torch.save(state, ckpt_path)
                total_acc, total_count = 0, 0
        print("\nTraining set: Average loss: {:.4f}\tAccuracy: {:.2f}%\n".format(
            np.mean(train_loss), np.mean(train_acc))
        )
        self.losses["train"].append(np.mean(train_loss))
        self.accuracy["train"].append(np.mean(train_acc))

    def on_epoch_valid(self):
        model_path = f"logs/saved_models/{self.model.__class__.__qualname__.lower()}.pth"
        model_state = {"model_state": self.model.state_dict()}

        self.model.eval()
        val_loss = []
        val_acc = []
        total_acc = 0.
        epoch_loss = 0.

        with torch.no_grad():
            for step, (image, targets) in enumerate(self.val_loader):
                batches_done = targets.size(0)
                image, targets = image.to(self.device), targets.to(self.device)
                pred = self.model(image)
                loss = self.loss_fn(pred, targets)
                epoch_loss += loss.item()
                pred = pred.argmax(dim=1, keepdim=True)
                total_acc += (pred == targets.view_as(pred)).sum().item()
                acc = 100 * total_acc / len(self.val_loader) / batches_done
            epoch_loss /= len(self.val_loader)
            val_loss.append(epoch_loss)
            val_acc.append(acc)

        print("\nValidation set: Average loss: {:.4f}\tAccuracy: {:.2f}%\n".format(
            epoch_loss, acc)
        )
        if epoch_loss < self.best_loss:
            print("***** New optimal state found, saving state ... ")
            torch.save(model_state, model_path)
        self.losses["valid"].append(np.mean(val_loss))
        self.accuracy["valid"].append(np.mean(val_acc))

    def show_batch(self):
        for images, labels in self.train_loader:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xticks([]), ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
            plt.savefig(f"logs/plots/train_sample.png", dpi=600, bbox_inches="tight")
            break
        if self.verbose:
            plt.show()

    @staticmethod
    def load_history(path="logs/history", metric="Losses"):
        hist_path = list(filter(lambda x: metric.lower() in x, os.listdir(path)))

        for p in hist_path:
            if p.startswith(metric.lower()):
                path = os.path.join(path, p)

        history = np.load(path)
        train_hist = history["train"]
        val_hist = history["valid"]
        return train_hist, val_hist

    @staticmethod
    def plot_history(save_dir="logs/plots", metric="losses", verbose=None):
        """Returns the plots of the training and validation of specified metrics.
        Plots of loss is returned if metric is `loss` or plots of accuracy is returned
        if metrics is `acc`.

        Parameters:
            save_dir: (str) Directory to save output plot. Default=`logs/plots`
            metric: (str) metric to monitor. Default is "loss".
            verbose: (bool) to display the plot. Default is None.
        """
        train_metric, val_metric = Trainer.load_history(metric=metric)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # axes parameters
        font = {"family": "sans-serif", "color": "black", "weight": "normal", "size": 13}

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(train_metric)), train_metric, color="#ffa500", label="Training")
        plt.plot(range(len(val_metric)), val_metric, color="green", label="Validation")
        plt.ylabel(f"{metric}", fontdict=font)
        plt.xlabel("Epochs", fontdict=font)
        plt.xlim([0, len(train_metric)])
        plt.legend(loc="best", frameon=False, prop={"size": 12})
        if verbose:
            plt.savefig(f"{save_dir}/{metric}.png", bbox_inches="tight", dpi=300)
        plt.show(block=False)

    def create_dir(self, directory):
        if not os.path.exists(directory):
            return os.makedirs(directory, exist_ok=True)


def main():
    device = torch.device("cuda")
    model = FerNet54()
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print("Training FERNet54 with CrossEntropy Loss")
    trainer = Trainer(model, optimizer, loss_fn, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                      epochs=NUM_EPOCHS, device=device, train_dir="data/train", val_dir="data/val",
                      verbose=True, logs_dir="logs", log_interval=200)

    # visualize training dataset and close the figure to start training
    trainer.show_batch()
    trainer.fit()
    trainer.plot_history(metric="Loss", verbose=True)
    trainer.plot_history(metric="Acc", verbose=True)


if __name__ == "__main__":
    main()
