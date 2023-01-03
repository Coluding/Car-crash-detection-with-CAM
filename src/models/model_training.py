import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))

from vgg19 import VGG19
from efficient_net import EfficientNet
import torch


def main():
    model = EfficientNet(new_model=True, model_to_load=r"C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\src\models\saved_models\EfficientNet\141222epoch_num50.model")
    model.fit(save_every_n_epoch=10, patience=5, factor=0.1) # Learning rate scheduler kwargs
    model.torch_save_model()
    model.save_model()
    #model = torch.load(r"C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\src\models\saved_models\EfficientNet\020123.test")

    print("hrllo")


if __name__ == "__main__":

    main()