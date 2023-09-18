import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

if __name__ == "__main__":

    response = requests.post("http://127.0.0.1:8000/predict",
                         files={'file': open(
                             r"C:\Users\lbierling\Downloads\crash1.jfif",
                                "rb")})

    # Check if the request was successful
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()