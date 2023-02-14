from PIL import Image
import io
import yaml
from ..model_architecture.final_model import FinalModel


def init():
    global model
    with open(r"../model_architecture/training/config.yml") as f:
        config = yaml.safe_load(f)
    model_path = config["specific_model_name_to_use"]
    data_path = config["create_train_test_dir"]["destination_path"]

    model = FinalModel(model_path, data_path)


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