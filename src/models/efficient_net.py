import torch
import torchinfo
import torchvision.transforms as tt
import torchvision
import pickle
import os
from src.models.base_model import BaseModel
from src.models.utils import ImageStats


class EfficientNet(BaseModel):
    def __init__(self):
        super().__init__()
        self._specific_config_file = self._config_file["efficient_net"]

        if self._config_file["create_train_test_dir"]["create_new_dirs"]:
            self.preprocess_images()

        self.name = "vgg19"
        self._collect_hyperparams()
        self._init_transforms() # set the transform attributes
        self._set_up_model() # combine classifer and backbone model
        self._load_images() # load images into dataloader

    def _init_transforms(self):
        """
        Get the transforms of the backbone model and add some more to it.  Compute the mean and std of the images to
        normalize the data

        :return: None
        """
        image_stats_train = ImageStats(os.path.join(
            self._config_file["create_train_test_dir"]["destination_path"], "train"))
        stats = image_stats_train.compute_stats() # Normalize image data with train data stats, so the algorithm gets no
        # information about the test data set

        self.train_transforms = tt.Compose([
            tt.Resize((255,255), interpolation=tt.InterpolationMode.BILINEAR),
            tt.RandomCrop((240,240)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

        self.val_transforms = tt.Compose([
            tt.Resize((255, 255), interpolation=tt.InterpolationMode.BILINEAR),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("saved_models/vgg19.model")
        else:
            weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
            model = torchvision.models.efficientnet_b1(weights)
            self.transforms = weights.transforms()

        for param in model.parameters():
            param.requires_grad = self.train_backbone_weights
        return model

    def _add_classifier(self):
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

        model = self._init_backbone_model()

        classifier = torch.nn.Sequential(*layers)

        model.classifier = classifier
        self.model = model

