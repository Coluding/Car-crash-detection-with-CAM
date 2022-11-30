from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from resnet import ResNetVanilla
from vgg19 import VGG19Vanilla
from custom_model import CustomModel1
import torch


def main():
    model = VGG19Vanilla()
    model.fit(patience=7, factor=0.5)
    inp = input("Save?")
    if inp == "yes":
        model.save_model()


if __name__ == "__main__":
    main()