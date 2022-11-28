import torch
import torchinfo
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
import numpy as np
import os
from pytorch_lightning import Trainer
from src.training.models.base_model_lightning import BaseModelLightning
from src.training.models.base_model import BaseModel


class VGG19Lightning(BaseModelLightning):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["vgg19"]
        self.name = "vgg19"
        self._collect_hyperparams()
        self._set_up_model()
        self._preprocess_images()
        for param in self.parameters():
            print(param.requires_grad)

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("saved_models/vgg19.model")
        else:
            weights = torchvision.models.VGG19_Weights.DEFAULT
            self.model = torchvision.models.vgg19(weights)
            self.transforms = weights.transforms()

        for param in self.model.parameters():
            param.requires_grad = self.train_backbone_weights

    def _add_classifier(self):
        print(self.classifier_layer)
        layers = self._create_layers(self.classifier_layer)

        self.hparams_dict["classifier_layer"] = layers

        classifier1 = torch.nn.Sequential(*layers)

        self.model.classifier = classifier1

    def save_model(self):
        if os.path.exists("saved_models/vgg19.model"):
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("saved_models")
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)


class VGG19Vanilla(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["vgg19"]
        self.name = "vgg19"
        self._collect_hyperparams()
        self._set_up_model()
        self._load_images()

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("saved_models/vgg19.model")
        else:
            weights = torchvision.models.VGG19_Weights.DEFAULT
            self.model = torchvision.models.vgg19(weights)
            self.transforms = weights.transforms()

        for param in self.model.parameters():
            param.requires_grad = self.train_backbone_weights

        print(torchinfo.summary(self.model, input_size=(32, 3, 256, 256)))

    def _add_classifier(self):
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

        classifier1 = torch.nn.Sequential(*layers)

        self.model.classifier = classifier1

    def save_model(self):
        if os.path.exists("saved_models/vgg19.model"):
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("saved_models")
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
