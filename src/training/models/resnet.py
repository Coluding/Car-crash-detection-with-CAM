import torch
import torchinfo
import torchvision
import pickle
import os
from src.training.models.base_model_lightning import BaseModelLightning
from src.training.models.base_model import BaseModel
from src.training.models.utils import ImageStats
import torchvision.transforms as tt


class ResNetLightning(BaseModelLightning):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["resnet"]
        self.name = "resnet"
        self._collect_hyperparams()
        self._set_up_model()
        self._preprocess_images()


    def _init_transforms(self):
        image_stats = ImageStats()
        stats = image_stats.compute_stats()
        self.transforms = tt.Compose([
            tt.Resize((256,256)),
            tt.RandomCrop((224,224)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

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
        print(self.transforms)

    def _init_transforms(self):
        image_stats = ImageStats()
        stats = image_stats.compute_stats()
        self.transforms = tt.Compose([
            tt.Resize((256,256)),
            tt.RandomCrop((224,224)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])
        # TODO: different tranforms for val data

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("saved_models/vgg19.model")
        else:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.model = torchvision.models.resnet18(weights)
            self.transforms = weights.transforms()

        for param in self.model.parameters():
            param.requires_grad = self.train_backbone_weights


    def _add_classifier(self):
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

        classifier1 = torch.nn.Sequential(*layers)

        self.model.fc = classifier1

    def save_model(self):
        if os.path.exists("saved_models/resnet/resnet.model"):
            with open("saved_models/resnet.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("saved_models/resnet")
            with open("saved_models/resnet/resnet.model", "wb") as f:
                pickle.dump(self, f)
