import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch.nn as nn
import yaml
import torch.nn.functional as F
import torchvision
import pickle
import datetime
from PIL import Image
import torchinfo
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopper


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        with open("../../config.yml") as y:
            self._config_file = yaml.safe_load(y)

        self.lightning = None
        self.today = datetime.datetime.today().strftime("Y-m-d")
        self.tb = SummaryWriter()
        self._specific_config_file = None
        self.transforms = None
        self.model = None
        self.classes = None
        self.train_loader = None
        self.val_loader = None
        self.val_accuracy = None
        self.val_loss = None
        self.history = []

    def _collect_hyperparams(self):
        self.epochs = self._specific_config_file["epochs"]
        self.batch_size = self._specific_config_file["batch_size"]
        self.learning_rate = self._specific_config_file["learning_rate"]
        self.train_backbone_weights = self._specific_config_file["train_backbone_params"]
        self.train_test_split_ratio = self._specific_config_file["train_test_split_ratio"]
        self.classifier_layer = self._specific_config_file["classifier_layer"]

        self.hparams_dict = {"epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                             "train_backbone_weights": self.train_backbone_weights,
                             "tran_test_split_ratio": self.train_test_split_ratio}

    def _create_layers(self, config_list_of_layers):
        final_layer_list = []
        layer_dict = {}
        for index, layer in enumerate(config_list_of_layers):
            in_shape = list(layer.values())[0][0]
            out_shape = list(layer.values())[0][1]
            dropout_rate = list(layer.values())[0][2]

            nn_layer = nn.Linear(in_features=in_shape, out_features=out_shape)
            final_layer_list.append(nn_layer)
            layer_dict[f"layer_{index}_shape_in"] = in_shape
            layer_dict[f"layer_{index}_shape_out"] = out_shape

            if dropout_rate != 0:
                dropout_layer = nn.Dropout(dropout_rate)
                final_layer_list.append(dropout_layer)
                layer_dict[f"layer_{index}_dropoutRrate"] = dropout_rate

            if index != len(config_list_of_layers):
                final_layer_list.append(nn.ReLU())

        return final_layer_list, layer_dict


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
                           tt.RandomRotation(degrees=30),
                           #tt.RandomCrop(size=190),
                           self.transforms])
        images = ImageFolder(path, transform=trfs)
        self.classes = images.classes
        # more augmenting !!
        split_ratio = 0.8
        train_images, test_images = random_split(images, [round(split_ratio * len(images)),
                                                          round((1 - split_ratio) * len(images))])

        train_loader = DataLoader(train_images, batch_size=self._specific_config_file["batch_size"], num_workers=2,
                                  shuffle=True)
        val_loader = DataLoader(test_images, batch_size=self._specific_config_file["batch_size"], num_workers=2,
                                shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def print_summary(self, input_size):
        print(torchinfo.summary(self.model, input_size=input_size))

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
        acc = self.accuracy(out, labels)
        return loss, acc

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

    def fit(self, optim=torch.optim.Adam, lrs=torch.optim.lr_scheduler.ReduceLROnPlateau, **kwargs):
        """
        Fits the model with specified hyperparams of the config file. If a learning rate scheduler is used, the **kwargs
        can be used to give the scheduler arguments
        :param lrs: Pytorch learning rate scheduler
        :param optim: Pytorch optimizer
        :param kwargs: additonal keyword arguments for the learning rate scheduler
        :return: None
        """
        optimizer = optim(self.parameters(), lr=self.learning_rate)
        lrs = lrs(optimizer, **kwargs)
        print(type(lrs))
        early_stopper = EarlyStopper(patience=2, min_delta=0.2)
        max_acc = 0

        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            train_accs = []
            for batch_number, batch in enumerate(self.train_loader):
                print(f"new batch: {batch_number}")
                loss, train_acc = self.training_step(batch)
                loss.backward()
                train_losses.append(loss)
                train_accs.append(train_acc)

                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1} done!")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            result = self.evaluate()

            result["learning_rate"] = optimizer.param_groups[0]["lr"]
            result["train_loss"] = torch.stack(train_losses).mean().item()
            result["train_acc"] = torch.stack(train_accs).mean().item()

            self.tb.add_scalar("Train loss", result["train_loss"], epoch)
            self.tb.add_scalar("Train accuracy", result["train_acc"], epoch)
            self.tb.add_scalar("Val accuracy", result["val_acc"], epoch)
            self.tb.add_scalar("Val loss", result["val_loss"], epoch)
            self.tb.add_scalar("Learning rate", result["learning_rate"], epoch)
            self._epoch_end_val(epoch + 1, result)
            self.history.append(result)

            lrs.step(metrics=result["val_loss"])

            if max_acc < result["val_acc"]:
                max_acc = result["val_acc"]

            if early_stopper.early_stop(result["val_loss"]):
                break
            else:
                print("No early stopping!")

        self.tb.add_hparams(self.hparams_dict, {"hparam/max_accuracy": max_acc})
        self.tb.close()
