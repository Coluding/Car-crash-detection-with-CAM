import os
from src.models import FinalModel
from src.models import custom_model, resnet, vgg19
import sys


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
    print(os.path.dirname(__file__), "models")
    os.chdir("models")
    m = FinalModel()
    print(m.model.history)


if __name__ == "__main__":
    main()