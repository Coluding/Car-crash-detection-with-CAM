import io
import os
import sys
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from PIL import Image
from src.models import FinalModel
from src.models import custom_model, resnet, vgg19


dir_name = os.path.basename(os.path.normpath(os.getcwd()))
print(dir_name)
print(".............................................")
print(os.getcwd())

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

    return {"prediction_name": out[0], "prediction_class": out[1]}


