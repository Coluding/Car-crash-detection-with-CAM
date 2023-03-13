import requests


if __name__ == "__main__":
    resp = requests.post("http://127.0.0.1:8000/predict",
                         files={'file': open(
                             r"C:\Users\lbierling\Downloads\crash3.jpg",
                             "rb")})
    print(resp.json())