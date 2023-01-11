import io
import os
import sys
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from PIL import Image
sys.path.append(os.path.abspath("./src/models"))

from utils import *
from transforms import ImageTransforms
from final_model import FinalModel
from efficient_net import EfficientNet


dir_name = os.path.basename(os.path.normpath(os.getcwd()))

if dir_name == "src":
    rel_path = "models"
else:
    rel_path = "src/models"


module_path = os.path.abspath(rel_path)
sys.path.append(module_path)
print(sys.path)


os.chdir(rel_path)
model = FinalModel()

app = FastAPI()


@app.post("/predict", status_code=status.HTTP_201_CREATED)
def predict(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        out = model.predict_raw_image(image)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="there was a error computing the prediction")
    finally:
        file.file.close()

    return {"prediction_class": out}


