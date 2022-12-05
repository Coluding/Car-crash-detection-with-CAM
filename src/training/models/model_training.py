import datetime

from vgg19 import VGG19Vanilla


def main():
    model = VGG19Vanilla()
    model.fit(patience=7, factor=0.1)
    today = datetime.datetime.today().strftime(fmt="%d%m%y")
    inp = input("Save?")
    if inp == "yes":
        model.save_model(today)


if __name__ == "__main__":
    main()