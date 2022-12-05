from flask import Flask
import requests


if __name__ == "__main__":
    resp = requests.post("http://localhost:5200/predict",
                         files={'file': open(r'D:\ML\DL\projects\insurance_image_recog\data2\Hood\0163.JPEG', 'rb')})
    print(resp.text)