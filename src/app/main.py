from flask import Flask, request, jsonify, render_template, redirect, flash
from src.training.models import FinalModel
from src.training.models import vgg19
from src.training.models import resnet
from src.training.models import custom_model
import sys
from PIL import Image


sys.modules["vgg19"] = vgg19
sys.modules["resnet"] = resnet
sys.modules["custom_model"] = custom_model
model = FinalModel()


app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config["UPLOAD_FOLDER"] = "/upload_folder"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("post")

        file = request.files["file"]

        if file is None or file.filename == "":
           return jsonify({'error': 'no file'})

        if not allowed_file(file.filename):
           return jsonify({'error': 'format not supported'})

        try:
            img = Image.open(file)
            prediction = model.predict_raw_image(img)[1]
            data = {'prediction': prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': f'error during prediction: {e}'})


if __name__ == "__main__":
    app.run(port=5200, debug=False)