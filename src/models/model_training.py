from vgg19 import VGG19Vanilla
from efficient_net import EfficientNet


def main():
    model = VGG19Vanilla()
    model.fit(save_every_n_epoch=10, patience=5, factor=0.1) # Learning rate scheduler kwargs
    inp = input("Save?")
    if inp == "yes":
        model.save_model()


if __name__ == "__main__":
    main()