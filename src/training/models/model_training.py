from pytorch_lightning import loggers
from pytorch_lightning import Trainer
from resnet import ResNetVanilla
from vgg19 import VGG19Vanilla
import torch


def main():
    model = VGG19Vanilla()
    model.fit(optim=torch.optim.Adam, lrs=torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", patience=2,  verbose=True)


if __name__ == "__main__":
    main()