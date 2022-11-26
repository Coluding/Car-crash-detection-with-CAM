from src.training.models.vgg19 import VGG19
import os
import pickle
import yaml
import torch


class FinalModel:
    def __init__(self, m):
        with open("../../config.yml") as f:
            self._config = yaml.safe_load(f)
        path = os.path.join("../../training/models/saved_models", self._config["model_to_use"] + ".model")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.m = m

    def preprocess_image(self, image):
        transforms = self.model.transforms
        transformed_image = transforms(image)
        final_image = torch.unsqueeze(transformed_image, 0)
        return final_image

    def predict_raw_image(self, image):
        transformed_image = self.preprocess_image(image)
        out = self.m(transformed_image)
        return out




