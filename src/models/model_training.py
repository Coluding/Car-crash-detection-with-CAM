from vgg19 import VGG19Vanilla


def main():
    model = VGG19Vanilla()
    model.fit(save_every_n_epoch=10, patience=3, factor=0.1)
    inp = input("Save?")
    if inp == "yes":
        model.save_model()


if __name__ == "__main__":
    main()