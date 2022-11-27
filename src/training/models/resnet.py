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
from src.training.models.base_model_lightning import BaseModel
from pytorch_lightning import loggers


class ResNet(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["resnet"]
        self._collect_hyperparams()
        self._set_up_model()
        self._preprocess_images()
        for param in self.parameters():
            print(param.requires_grad)

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("saved_models/vgg19.model")
        else:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.model = torchvision.models.resnet18(weights)
            self.transforms = weights.transforms()

        for param in self.model.parameters():
            param.requires_grad = self._specific_config_file["freeze_backbone_params"]

        print(self.model.fc)
        print(torchinfo.summary(self.model, input_size=(32, 3, 256, 256)))

    def _add_classifier(self):
        layers = self._create_layers(self.classifier_layer)

        self.hparams_dict["classifier_layer"] = layers

        classifier1 = torch.nn.Sequential(*layers)

        self.model.fc = classifier1

    def save_model(self):
        if os.path.exists("saved_models/vgg19.model"):
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("saved_models")
            with open("saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)


if __name__ == "__main__":
    model = ResNet()
    print(model.transforms)
    logger = loggers.TensorBoardLogger("tb_logger", name="resnet")
    #print(torchinfo.summary(model.model, input_size=(32,3,256,256)))
    #print(model.model.classifier )
    # logger.log_hyperparams(model.hparams_dict)
    trainer = Trainer(max_epochs=model.epochs, logger=logger, log_every_n_steps=8)
    trainer.fit(model)
    print("ss")