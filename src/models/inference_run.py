import json
import sys
import os
from PIL import Image
import io
from transforms import ImageTransforms
from final_model import FinalModel


def init():
    global model
    model = FinalModel()


def run(file):
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        out = model.predict_raw_image(image)
    except Exception:
        raise TypeError("Could not convert image to tensor!")
    finally:
        file.file.close()

    return {"prediction_name": out[0], "prediction_class": out[1]}


init()