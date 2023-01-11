import os.path
import pickle
import yaml
import torch
from PIL import Image
import sys
from utils import *
from transforms import ImageTransforms


class FinalModel:
    def __init__(self):
        with open(r"config.yml") as f:
            self._config = yaml.safe_load(f)

        self._path = self._config["specific_model_name_to_use"]
        self._destination_path = self._config["create_train_test_dir"]["destination_path"]
        #with open(path, "rb") as f:
        #    self.model = pickle.load(f)
        try:
            self.model = torch.load(self._path)
        except RuntimeError:
            self.model = torch.load(self._path, map_location=torch.device('cpu'))

        self.val_transforms = None
        self.train_transforms = None

        self._set_transforms()

    def _set_transforms(self):
        transforms = ImageTransforms(self._destination_path)
        if "efficientnet" in self._path.lower():
            self.train_transforms = transforms.efficient_net_train_transforms
            self.val_transforms = transforms.efficient_net_val_transforms

        elif "vgg19" in self._path.lower():
            self.train_transforms = transforms.vgg19_train_transforms
            self.val_transforms = transforms.vgg19_val_transforms

    def preprocess_image(self, image):
        transformed_image = self.val_transforms(image)
        final_image = torch.unsqueeze(transformed_image, 0)
        return final_image

    def predict_raw_image(self, image):
        transformed_image = self.preprocess_image(image)
        out = self.model(transformed_image)
        out = torch.max(out, dim=1)[1].item()
        #out_class = self.model.classes[out]
        return out


if __name__ == "__main__":
    f = FinalModel()
    img = Image.open(r"C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\data3\train\crash\38.jpg")
    print(f.predict_raw_image(img))
