import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from torchvision.datasets import ImageFolder
import yaml
from torch.utils.data import DataLoader, random_split


class BaseModel(ABC, pl.LightningModule):
    def __init__(self):
        super().__init__()

        with open("../../config.yml") as y:
            self._config_file = yaml.safe_load(y)

        self._specific_config_file = None
        self.transforms = None
        self.model = None
        self.classes = None
        self.val_accuracy = None
        self.val_loss = None
        self.history = []
        self.train_images = None
        self.test_images = None

        self.batch_size = None

    @abstractmethod
    def _init_backbone_model(self, new_model=True):
        pass

    @abstractmethod
    def _add_classifier(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @staticmethod
    def accuracy(output, labels):
        _, preds = torch.max(output, dim=1)
        return torch.tensor(torch.sum(preds == labels) / len(preds))

    def _set_up_model(self):
        self._init_backbone_model()
        self._add_classifier()

    def _create_layers(self, config_list_of_layers):
        final_layer_list = []
        for index, layer in enumerate(config_list_of_layers):
            print(layer)
            nn_layer = nn.Linear(in_features=list(layer.values())[0][0], out_features=list(layer.values())[0][1])
            final_layer_list.append(nn_layer)

            if list(layer.values())[0][2] != 0:
                dropout_layer = nn.Dropout(list(layer.values())[0][2])
                final_layer_list.append(dropout_layer)

            if index != len(config_list_of_layers):
                final_layer_list.append(nn.ReLU())

        return final_layer_list

    def _collect_hyperparams(self):
        self.epochs = self._specific_config_file["epochs"]
        self.batch_size = self._specific_config_file["batch_size"]
        self.learning_rate = self._specific_config_file["learning_rate"]
        self.train_backbone_weights = self._specific_config_file["freeze_backbone_params"]
        self.train_test_split_ratio = self._specific_config_file["train_test_split_ratio"]
        self.classifier_layer = self._specific_config_file["classifier_layer"]

        self.hparams_dict = {"epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                             "train_backbone_weights": self.train_backbone_weights,
                             "tran_test_split_ratio": self.train_test_split_ratio}

    def forward(self, xb):
        out = self.model(xb)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._specific_config_file["learning_rate"])

    def training_step(self, batch, batch_isx):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        self.log('step_training_loss', loss, on_step=True, on_epoch=False)
        self.log('step_acc_loss', acc, on_step=True, on_epoch=False)

        return {"loss": loss, "acc": acc}

    def _preprocess_images(self):
        path = self._config_file["image_path"]
        trfs = tt.Compose([tt.RandomVerticalFlip(),
                           tt.RandomHorizontalFlip(),
                           tt.RandomRotation(degrees=30),
                           #tt.RandomCrop(size=190),
                           self.transforms])
        dataset = ImageFolder(path, transform=trfs)
        split_ratio = 0.8
        self.classes = dataset.classes
        self.train_images, self.test_images = random_split(dataset, [round(split_ratio * len(dataset)),
                                                           round((1 - split_ratio) * len(dataset))])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_images,
                                  batch_size=self._specific_config_file["batch_size"],
                                  shuffle=True, num_workers=2)

        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(dataset=self.test_images,
                                 batch_size=self._specific_config_file["batch_size"],
                                 shuffle=False, num_workers=2)

        return test_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        out = self(images)
        acc = self.accuracy(out,labels)
        loss = F.cross_entropy(out, labels)
        # tensorboard_logs = {'val_loss': loss, "val_acc": self.valid_acc}
        self.log("step_val_loss", loss, on_step=True, on_epoch=False)
        self.log("step_val_accuracy", acc, on_step=True, on_epoch=False)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        self.log('epoch_val_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('epoch_val_acc', avg_acc, on_step=False, on_epoch=True)

        return {"epoch_val_loss": avg_loss, "epoch_val_acc": avg_acc}

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log('epoch_training_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('epoch_training_acc', avg_acc, on_step=False, on_epoch=True)

        # tensorboard_logs = {"epoch_train_loss": avg_loss, "epoch_train_acc": avg_acc}
        # return {"loss": avg_loss, "log": tensorboard_logs}

