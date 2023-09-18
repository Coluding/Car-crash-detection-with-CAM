import io
import os
import sys
import cv2 as cv
import numpy as np
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO


from src.model_architecture.training.final_model import FinalModel


img_stats = "./model_architecture/training/config/image_stats.json"
p = r"C:\Users\lbierling\Documents\priv\projects\insurance_image_recog\src\model_architecture\training\saved_models\050123_f1_score=1.0epoch_num90.torch"
model = FinalModel(p, image_stats_path=img_stats)

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"], )


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        print("in try")
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        output = model.get_cam(image)
        output_with_cam = compute_and_return_cam(image, output)
        final_img = Image.fromarray(output_with_cam)

        img_io = BytesIO()
        final_img.save(img_io, 'JPEG')
        img_io.seek(0)
        print("hieleel")

        return StreamingResponse(img_io, media_type="image/jpeg")

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="there was a error computing the prediction")
    finally:
        file.file.close()


def compute_and_return_cam(img: Image, cam: np.ndarray, resize:int = 256):
    cam = np.clip(cam, 0, 255)
    img = np.array(img)
    img = cv.resize(img, (resize, resize))
    cam = cam.astype(np.uint8)
    cam_normalized = 255 - cv.normalize(cam, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    hm = cv.applyColorMap(cam_normalized, cv.COLORMAP_JET)
    a = 0.5
    b = 1 - a
    blended = cv.addWeighted(img, a, hm, b, 0)

    return blended