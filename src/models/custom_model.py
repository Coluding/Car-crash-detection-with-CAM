import torch
import pickle
import os
from src.models.base_model import BaseModel
import torch.nn as nn
import torchvision.transforms as tt
from src.models.utils import ImageStats


class CustomModel1(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["custom1"]
        self.name = "custom1"
        self._collect_hyperparams()
        self._init_transforms()
        self._set_up_model()
        self._load_images()
        for name, param in self.named_parameters():
            print(name, param.grad)

    def _init_transforms(self):
        image_stats = ImageStats()
        stats = image_stats.compute_stats()
        self.transforms = tt.Compose([
            tt.Resize((200,200)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

    def _init_backbone_model(self, new_model=True):
        activation_function = self._get_activation_function()
        if not new_model:
            self.model = torch.load("../saved_models/custom1.model")
        else:
            self.model = nn.Sequential(nn.Conv2d(3,64, (3,3)),
                                       activation_function,
                                       nn.Conv2d(64,64, (3,3)),
                                       activation_function,
                                       nn.MaxPool2d((2,2)),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(64,32,(3,3)),
                                       activation_function,
                                       nn.Conv2d(32,32,(3,3)),
                                       activation_function,
                                       nn.MaxPool2d((2,2)),
                                       nn.BatchNorm2d(32),
                                       nn.Conv2d(32,16,(3,3)),
                                       activation_function,
                                       nn.Conv2d(16,16, (3,3)),
                                       activation_function,
                                       nn.MaxPool2d((4,4)),
                                       nn.Flatten())


    def _add_classifier(self):
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

        classifier = torch.nn.Sequential(*layers)

        self.model.append(classifier)

    def save_model(self):
        if os.path.exists("../saved_models/vgg19.model"):
            with open("../saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
        else:
            os.mkdir("../saved_models")
            with open("../saved_models/vgg19.model", "wb") as f:
                pickle.dump(self, f)
