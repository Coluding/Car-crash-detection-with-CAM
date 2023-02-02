import torch
import torchinfo
import torchvision.transforms as tt
import torchvision
import pickle
import os
from base_model import BaseModel
from transforms import ImageTransforms



class EfficientNet(BaseModel):
    def __init__(self, new_model=True, model_to_load=None, remote_run=False, train_path=None, val_path=None):
        """
        Constructor of base model

        :param new_model: True, if a new model should be built and no previuous model should be loaded
        :type new_model: bool
        :param model_to_load: if new_model is false, then path to the model pickled model that should be loaded for fine tuning
        :type model_to_load: bool
        :param remote_run: True, if the model is trained in the cloud
        :type remote_run: bool
        :param train_path: Remote path of the data if the model is trained in the cloud
        :type train_path: str
        :param val_path: Remote path of the validation data if the model is trained in the cloud
        :type val_path: str:
        :return: None
        """
        super().__init__()
        self._specific_config_file = self._config_file["efficient_net"]
        self._train_remote = remote_run
        self.azure_train_path = train_path
        self.azure_val_path = val_path

        if self._config_file["create_train_test_dir"]["create_new_dirs"]:
            self.preprocess_images()

        self.name = "EfficientNet"

        if new_model:
            self._collect_hyperparams()
            self._init_transforms() # set the transform attributes
            self._set_up_model() # combine classifer and backbone model

        else:
            self._setup_presaved_model(model_to_load=model_to_load)

        if remote_run:
            self._load_images(train_path=train_path, val_path=val_path)  # load images into dataloader

        else:
            self._load_images()

    def _setup_presaved_model(self, model_to_load):
        model = super()._setup_presaved_model(model_to_load)

        if not isinstance(model, self.__class__):
            raise TypeError("The model you are trying to use is not a EfficientNet model!")

    def _init_transforms(self):
        """
        Get the transforms of the backbone model and add some more to it.  Compute the mean and std of the images to
        normalize the data

        :return: None
        """
        if self._train_remote:
            image_path = self.azure_train_path
        else:
            image_path = self._config_file["image_path"]

        transforms = ImageTransforms(image_path)

        self.train_transforms = transforms.efficient_net_train_transforms

        self.val_transforms = transforms.efficient_net_val_transforms

    def _init_backbone_model(self, new_model=True):
        if not new_model:
            self.model = torch.load("../saved_models/vgg19.model")
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

    def _get_class_weights(self):
        if self._train_remote:
            super()._get_class_weights(self.azure_train_path)
        else:
            super()._get_class_weights()

