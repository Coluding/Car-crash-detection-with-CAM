import requests


if __name__ == "__main__":
    resp = requests.post("http://127.0.0.1:8000/predict",
                         files={'file': open(r'D:\ML\DL\projects\insurance_image_recog\data2\Hood\0163.JPEG', 'rb')})
    print(resp.json())