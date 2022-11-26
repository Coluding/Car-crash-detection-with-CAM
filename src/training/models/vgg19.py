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
import numpy as np
import os
from pytorch_lightning import Trainer
from src.training.models.base_model_lightning import BaseModel
from pytorch_lightning import loggers


class VGG19(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["vgg19"]
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
            param.requires_grad = False
                #self._specific_config_file["freeze_backbone_params"]

    def _add_classifier(self):
        layers = self._create_layers(self.classifier_layer)

        self.hparams_dict["classifier_layer"] = layers

        classifier = torch.nn.Sequential(*layers)

        self.model.classifier = classifier

    def save_model(self):
        if os.path.exists("saved_models/vgg19.model"):
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("saved_models")
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)


if __name__ == "__main__":
    model = VGG19()

    logger = loggers.TensorBoardLogger("tb_logger", name="vgg19")
    logger.log_hyperparams(model.hparams_dict)
    trainer = Trainer(max_epochs=model.epochs, logger=logger, log_every_n_steps=8)
    trainer.fit(model)
    print("ss")