import io
import os
import sys
from fastapi import FastAPI, Response, HTTPException, status, File, UploadFile
from PIL import Image
from src.training.models import FinalModel
from src.training.models import vgg19
from src.training.models import resnet
from src.training.models import custom_model

os.chdir("src/training/models")
sys.modules["vgg19"] = vgg19
sys.modules["resnet"] = resnet
sys.modules["custom_model"] = custom_model
model = FinalModel()

app = FastAPI()

@app.post("/predict", status_code=status.HTTP_201_CREATED)
def predict(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        print(type(image_bytes))
        image = Image.open(io.BytesIO(image_bytes))
        out = model.predict_raw_image(image)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="there was a error computing the prediction")
    finally:
        file.file.close()

    return {"prediction_name": out[0], "prediction_class": out[1]}


