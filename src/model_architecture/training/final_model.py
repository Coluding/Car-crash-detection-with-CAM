import os.path
import torchvision.models
import yaml
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import torch

try:
    from insurance_image_recog.src.model_architecture.training.transforms import ImageTransforms
except ImportError:
    from training.transforms import ImageTransforms


class FinalModel:
    def __init__(self, model_path, data_path=None):
        """
        Construcor of the final model used for production

        :param model_path: path of the torch model
        :type model_path: str
        :param data_path: path of the image data used for creating random subplots of images
        :type data_path: str
        """

        self._path = model_path
        self._destination_path = data_path

        # try to load model on gpu, otherwise use cpu
        try:
            self.model = torch.load(self._path)
        except RuntimeError:
            self.model = torch.load(self._path, map_location=torch.device('cpu'))

        self._class_names = ["Crash", "Normal"] #TODO: Anpassung nötig auf usecase

        self.transforms = None
        self.val_transforms = None
        self.train_transforms = None
        self._current_prediction = None

        self._set_transforms()

    def sample_random_image(self) -> str:
        """
        Samples random image from the validation data by sampling the paths

        :return: path to random sampled image
        :rtype: str
        """

        if self._destination_path is None:
            raise ValueError("Cannot sample random image when destination path is null!")

        base_dir = "test"
        base_path = os.path.join(self._destination_path, base_dir)

        crash_or_normal = os.listdir(base_path)[np.random.choice(len(os.listdir(base_path)))]
        intermediate_path = os.path.join(base_path, crash_or_normal)

        image_file = os.listdir(intermediate_path)[np.random.choice(len(os.listdir(intermediate_path)))]
        final_image_file = os.path.join(intermediate_path, image_file)

        return final_image_file

    def _set_transforms(self) -> None:
        """
        Sets up the correct transforms specified by the used model

        :return: None
        """
        self.transforms = ImageTransforms(self._destination_path)
        if "efficientnet" in self._path.lower():
            self.train_transforms = self.transforms.efficient_net_train_transforms
            self.val_transforms = self.transforms.efficient_net_val_transforms

        elif "vgg19" in self._path.lower():
            self.train_transforms = self.transforms.vgg19_train_transforms
            self.val_transforms = self.transforms.vgg19_val_transforms
        else:
            #TODO welche transforms nehmen, wenn modell nicht mit name abgespeichert ist
            self.train_transforms = self.transforms.efficient_net_train_transforms
            self.val_transforms = self.transforms.efficient_net_val_transforms

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.tensor:
        """
        Preprocesses image with the specified transforms, such that it can be given as input for the model

        :param image: image to be transformed
        :type image: PIL.image or path to image
        :return: transformed image as tensor
        :rtype: torch.tensor
        """

        if isinstance(image, str):
            image = Image.open(image)

        transformed_image = self.val_transforms(image)
        final_image = torch.unsqueeze(transformed_image, 0)
        final_image_denormalized = self.transforms.denormalize_img(transformed_image, self.transforms.stats)
        return final_image, final_image_denormalized

    def predict_raw_image(self, image) -> int:
        """
        Predicts the class of a raw image

        :param image:
        :return:
        """
        transformed_image = self.preprocess_image(image)[0]
        out = self.model(transformed_image)
        out = torch.max(out, dim=1)[1].item()
        return out

    def get_class_activation_map(self, input) -> torch.tensor:
        """
        Computes the output of the last convolutional layer, e.g. the feature map for class activation maps
        That makes it easy to debug the network in case of wrong predictions. The class activation maps shows the areas
        of interest for the network by extracting, combining and upsampling the feature map of the last convolutional
        layer

        :param input: input image of for which the feature map should be computed
        :type input: PIL.Image
        :return: tensor of the class activation map
        :rtype: torch.tensor
        """

        input_shape = input.shape[-1]

        # instantiate the last convolutional layer depending on the model
        if isinstance(self.model, torchvision.models.EfficientNet):
            last_conv_layer = self.model.features[8]
        else:
            raise RuntimeError("The model is not an instance of a pytorch model that can be used in this class")

        outputs = []

        # instantiate hook for the last convolutional layer
        def hook(module, input, output):
            outputs.append(output)

        handle = last_conv_layer.register_forward_hook(hook)

        # compute output
        prediction = self.model(input)
        _, winning_class = torch.max(prediction, dim=1)
        self._current_prediction = self._class_names[winning_class]

        feature_map_dim = outputs[0].shape[2]
        # get the output of last conv layer, i.e. the feature map and transform it to a shape that makes matmul possible
        last_conv_layer_output = outputs[0]
        weights_for_winning_class = list(self.model.classifier.parameters())[0][winning_class]

        # Remove the hook
        handle.remove()

        # each feature map of the last convolutional layer is multiplied with the corresponding weight of the fully
        # connected layer of the winning class and added up (F0*w0 + F1*w1 + ..... + Fk*wk) to get a final tensor
        # with the dimension HxW of the feature map but weighted with the weights of the fully connected layer

        final_tensor_class_activation = torch.zeros(last_conv_layer_output.shape[-2:])
        for i in range(weights_for_winning_class.squeeze().shape[0]):
            with torch.no_grad():
                final_tensor_class_activation += last_conv_layer_output.squeeze()[i] *\
                                                 weights_for_winning_class.squeeze()[i]

        # upsample the image to get original size by extracting the factor needed to upsample it to the
        # same size as the input
        upsample_factor = (input_shape / last_conv_layer_output.shape[-1])
        class_activation_map = ndimage.zoom(final_tensor_class_activation, (upsample_factor, upsample_factor), order=1)

        return class_activation_map

    def plot_cam(self, img_path=None):
        """
        Plots the class activation map for the image

        :return: None
        """
        # get image path
        if not img_path:
            img_path = self.sample_random_image()

        img_transformed, img_transformed_denormalized = f.preprocess_image(img_path)
        cam = f.get_class_activation_map(img_transformed)

        # create plots
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # remove fourth dimension and resample so matplotlib can display the image
        ax1.imshow(img_transformed_denormalized.squeeze().permute(1,2,0))
        ax2.imshow(img_transformed_denormalized.squeeze().permute(1, 2, 0), alpha=0.9)
        ax2.imshow(cam, alpha=0.5, cmap="jet")
        ax1.set_title(f"Prediction: {self._current_prediction}")
        ax1.axis("off")
        ax2.axis("off")
        plt.show()


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    with open(r"config/config.yml") as f:
        config = yaml.safe_load(f)

        model_path = config["specific_model_name_to_use"]
        data_path = config["create_train_test_dir"]["destination_path"]

    f = FinalModel(model_path, data_path)
    f.plot_cam(r"C:\Users\lbierling\Downloads\crash3.jpg")
    while True:
        try:
            f.plot_cam()
        except RuntimeError as E:
            print(E)
            continue
