import os.path

import lightning_lite
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import pickle
import datetime
from PIL import Image
import torchinfo
from abc import ABC, abstractmethod
from src.utils import run_with_mlflow
from pytorch_lightning import Trainer
import pytorch_lightning as pl


class BaseModel(ABC, pl.LightningModule):
    def __init__(self):
        super().__init__()
        with open("../../config.yml") as y:
            self._config_file = yaml.safe_load(y)

        self.lightning = None
        self.today = datetime.datetime.today().strftime("Y-m-d")
        self._specific_config_file = None
        self.transforms = None
        self.model = None
        self.classes = None
        self.train_loader = None
        self.val_loader = None
        self.val_accuracy = None
        self.val_loss = None
        self.history = []

    def _lightning_setup(self):
        self.lightning = lightning_lite.LightningLite()

    @staticmethod
    def accuracy(output, labels):
        _, preds = torch.max(output, dim=1)
        return torch.tensor(torch.sum(preds == labels) / len(preds))

    def evaluate(self):
        with torch.no_grad():
            self.eval()
            outputs = [self.validation_step(batch) for batch in self.val_loader]
            return self.validation_epoch_end(outputs)

    def _load_images(self):
        path = self._config_file["image_path"]
        trfs = tt.Compose([tt.RandomVerticalFlip(),
                           tt.RandomHorizontalFlip(),
                           self.transforms])
        images = ImageFolder(path, transform=trfs)
        self.classes = images.classes
        # more augmenting !!
        split_ratio = 0.8
        train_images, test_images = random_split(images, [round(split_ratio * len(images)),
                                                          round((1 - split_ratio) * len(images))])

        train_loader = DataLoader(train_images, batch_size=self._specific_config_file["batch_size"], num_workers=2, shuffle=True)
        val_loader = DataLoader(test_images, batch_size=self._specific_config_file["batch_size"], num_workers=2, shuffle=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

    @abstractmethod
    def _init_backbone_model(self, new_model=True):
        pass

    @abstractmethod
    def _add_classifier(self):
        pass

    def _set_up_model(self):
        self._init_backbone_model()
        self._add_classifier()

    def forward(self, xb):
        out = self.model(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        accuracy = self.accuracy(out, labels)
        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_losses = torch.stack(batch_losses).mean().item()
        epoch_acc = torch.stack(batch_acc).mean().item()
        return {"val_loss": epoch_losses, "val_acc": epoch_acc}

    def _epoch_end_val(self, epoch, results):
        print(f"Epoch[{epoch}]: val_loss: {results['val_loss']}"
              f" val_acc: {results['val_acc']}")

    def fit(self, optimizer=torch.optim.Adam):

        optim = optimizer(self.parameters(), lr=self._learning_rate)

        for epoch in range(self._specific_config_file["epochs"]):
            self.train()
            train_losses = []
            for batch_number, batch in enumerate(self.train_loader):
                print(f"new batch: {batch_number}")
                loss = self.training_step(batch)
                loss.backward()
                train_losses.append(loss)

                optim.step()
                optim.zero_grad()

            print(f"Epoch {epoch + 1} done!")
            result = self.evaluate()
            result["train_loss"] = torch.stack(train_losses).mean().item()
            self._epoch_end_val(epoch + 1, result)
            self.history.append(result)

    def log_model_mlflow(self):
        run_with_mlflow(self)

    @abstractmethod
    def save_model(self):
        pass