import torch
from src.model_architecture.training.final_model import FinalModel
from PIL import Image
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    p = r"C:\Users\lbierling\Documents\priv\projects\insurance_image_recog\src\model_architecture\training\saved_models\050123_f1_score=1.0epoch_num90.torch"
    img_stats = "../training/config/image_stats.json"
    m = FinalModel(p, image_stats_path=img_stats)

    img = Image.open(r"C:\Users\lbierling\Downloads\crash1.jfif")
    cam = m.get_cam(img)
    cam = np.clip(cam,0,255)
    img = np.array(img)

    img = cv.resize(img,(256,256))

    cam = cam.astype(np.uint8)
    cam_normalized = 255 - cv.normalize(cam, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    hm = cv.applyColorMap(cam_normalized, cv.COLORMAP_JET)

    a = 0.5
    b = 1 - a
    comb = cv.addWeighted(img, a, hm, b, 0)

    plt.imshow(comb)
    plt.show()

if __name__ == "__main__":
    main()
