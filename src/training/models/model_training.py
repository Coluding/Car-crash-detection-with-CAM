from vgg19 import VGG19Vanilla


def main():
    model = VGG19Vanilla()
    model.fit(patience=7, factor=0.1)
    inp = input("Save?")
    if inp == "yes":
        model.save_model("vgg19_051222")


if __name__ == "__main__":
    main()