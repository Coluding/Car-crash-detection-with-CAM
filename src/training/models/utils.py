import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import yaml
import torchvision.transforms as tt


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ImageStats:
    def __init__(self):
        self.image_data = None
        self._load_images()

    def _load_images(self):
        with open("../../config.yml") as y:
            config = yaml.safe_load(y)

        transforms = tt.Compose([tt.Resize((200, 200)),
                                 tt.ToTensor()])

        path_to_data = config["image_path"]
        images = ImageFolder(path_to_data, transforms)

        self.image_data = DataLoader(images, batch_size=16)

    def compute_stats(self):
        all_means = torch.zeros((len(self.image_data), 3))
        all_stds = torch.zeros((len(self.image_data), 3))
        for num, batch in enumerate(self.image_data):
            data, label = batch
            for channel in range(3):
                all_means[num][channel] = torch.mean(data[:, channel, :, :]).item()
                all_stds[num][channel] = torch.std(data[:, channel, :, :]).item()

        return torch.mean(all_means, dim=0), torch.mean(all_stds, dim=0)
