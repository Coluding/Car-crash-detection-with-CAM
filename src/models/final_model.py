import pickle
import yaml
from src.models.utils import *


class FinalModel:
    def __init__(self):
        with open(r"../config.yml") as f:
            self._config = yaml.safe_load(f)

        #path = fr"./saved_models/{self._config['model_to_use']}/{self._config['model_to_use']}.model"
        path = self._config["specific_modelName_to_use"]
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_image(self, image):
        transforms = self.model.val_transforms
        transformed_image = transforms(image)
        final_image = torch.unsqueeze(transformed_image, 0)
        return final_image

    def predict_raw_image(self, image):
        transformed_image = self.preprocess_image(image)
        out = self.model(transformed_image)
        out = torch.max(out, dim=1)[1].item()
        out_class = self.model.classes[out]
        return out, out_class

