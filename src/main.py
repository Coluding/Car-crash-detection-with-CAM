import os

from src.training.models import FinalModel
from src.training.models import vgg19
from src.training.models import resnet
from src.training.models import custom_model
import sys


def main():
    os.chdir("./training/models")
    sys.modules["vgg19"] = vgg19
    sys.modules["resnet"] = resnet
    sys.modules["custom_model"] = custom_model
    m = FinalModel()
    print(m.model.history)


if __name__ == "__main__":
    main()