from src.training.models.vgg19 import VGG19
from final_model import FinalModel
import os
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    os.chdir("training/models")
    m = FinalModel()
    print(m.model.history)
    for batch in m.model.val_loader:
        image, label = batch
        out = m.model(image)
        print(m.model.accuracy(out, label))
    for batch in m.model.train_loader:
        image, label = batch
        out = m.model(image)
        print(m.model.accuracy(out, label))


if __name__ == "__main__":
    os.chdir("training/models")
    m = FinalModel(VGG19().load_from_checkpoint(r"C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog\src\training\models\lightning_logs\version_1\checkpoints\epoch=1-step=20.ckpt"))
    img = Image.open(r"C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog\data\Bumper Front\0082.JPEG")
    print(F.softmax(m.predict_raw_image(img)))
