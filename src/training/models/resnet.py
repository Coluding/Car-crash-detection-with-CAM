import torch
import torchinfo
import torchvision
import pickle
import os
from src.training.models.base_model_lightning import BaseModelLightning
from src.training.models.base_model import BaseModel


class ResNetLightning(BaseModelLightning):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["resnet"]
        self.name = "resnet"
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
            param.requires_grad = self.train_backbone_weights

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


class ResNetVanilla(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["resnet"]
        self.name = "resnet"
        self._collect_hyperparams()
        self._set_up_model()
        self._load_images()
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
            param.requires_grad = self.train_backbone_weights

        print(torchinfo.summary(self.model, input_size=(32, 3, 256, 256)))

    def _add_classifier(self):
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

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
