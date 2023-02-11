import requests


if __name__ == "__main__":
    resp = requests.post("http://127.0.0.1:8000/predict",
                         files={'file': open(
                             r"D:\ML\DL\projects\insurance_image_recog\data4\test\normal\Acura_RDX_2011_34_18_240_23_4_73_65_180_17_AWD_5_4_SUV_Gcy.jpg",
                             'rb')})
    print(resp.json())