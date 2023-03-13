import io
import os
import sys
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from PIL import Image

#TODO: Wie mit transforms umgehen? Wie sollen die reingeladen werden

dir_name = os.path.basename(os.path.normpath(os.getcwd()))
if dir_name == "src":
    rel_path = "model_architecture"
    sys.path.append(os.path.abspath("./model_architecture"))
    sys.path.append(os.path.abspath("./model_architecture/training"))
else:
    rel_path = "../../model_architecture"
    sys.path.append(os.path.abspath(rel_path))
    sys.path.append(os.path.join(os.path.abspath(rel_path), "training"))

from final_model import FinalModel

module_path = os.path.abspath(rel_path)
sys.path.append(module_path)


os.chdir(os.path.join(os.path.abspath(rel_path), "training"))
model = FinalModel(r'C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\src\model_architecture\56_1.0_best_model_mlflow\data\model.pth')

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


