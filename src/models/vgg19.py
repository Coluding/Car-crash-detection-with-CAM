import torch
import torchvision.transforms as tt
import torchvision
import pickle
import os
from base_model import BaseModel
from transforms import ImageTransforms


class VGG19(BaseModel):
    def __init__(self, new_model=True, model_to_load=None):
        """
        Constructor

        :param new_model: True, if new model should be built and trained
        :type new_model: bool
        :param model_to_load: Path of saved to load
        :type model_to_load: str
        :return: None
        """
        super().__init__()
        self._specific_config_file = self._config_file["vgg19"]

        if self._config_file["create_train_test_dir"]["create_new_dirs"]:
            self.preprocess_images()

        self.name = "vgg19"
        if new_model:
            self._collect_hyperparams()
            self._init_transforms() # set the transform attributes
            self._set_up_model() # combine classifer and backbone model

        else:
            self._setup_presaved_model(model_to_load=model_to_load)

        self._load_images()  # load images into dataloader

    def _setup_presaved_model(self, model_to_load):
        model = super()._setup_presaved_model(model_to_load)

        if not isinstance(model, self.__class__):
            raise TypeError("The model you are trying to use is not a VGG19 model!")

    def _init_transforms(self):
        """
        Get the transforms of the backbone model and add some more to it.  Compute the mean and std of the images to
        normalize the data

        :return: None
        """
        transforms = ImageTransforms()

        self.train_transforms = transforms.vgg19_train_transforms

        self.val_transforms = transforms.vgg19_val_transforms

    def _init_backbone_model(self, new_model=True):
        """
        Initializes pretrained backbone model

        :param new_model: True, if a new model should be trained
        :type new_model: bool
        :return: backbone model
        :rtype: torchvision.models
        """
        if not new_model:
            self.model = torch.load("../saved_models/vgg19.model")
        else:
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg19(weights)
            self.transforms = weights.transforms()

        for param in model.parameters():
            param.requires_grad = self.train_backbone_weights
        return model

    def _add_classifier(self):
        """
        Adds custom classifier to the end of the backbone model for training on the correct amount of classes and sets
        the self.model attribute

        :return: None
        """
        layers, layer_dict = self._create_layers(self.classifier_layer)

        self.hparams_dict = {**self.hparams_dict, **layer_dict}

        model = self._init_backbone_model()

        classifier = torch.nn.Sequential(*layers)

        model.classifier = classifier
        self.model = model


def main():
    with open("../saved_models/pickled_models/vgg19/vgg19.model", "rb") as m:
        model = pickle.load(m)

    test_data = next(iter(model.val_loader))[0]
    test_target = next(iter(model.val_loader))[1]
    pred = model(test_data)
    print(torch.max(pred, dim=1)[1])
    print(test_target)
    print(model.accuracy(pred, test_target))


if __name__ == "__main__":
    main()