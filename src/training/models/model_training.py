from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from resnet import ResNetVanilla
from vgg19 import VGG19Vanilla
from custom_model import CustomModel1
import torch


def main():
    model = ResNetVanilla()
    model.fit(patience=2)


if __name__ == "__main__":
    main()