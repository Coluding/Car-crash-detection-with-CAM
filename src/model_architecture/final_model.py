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
    from .training.transforms import ImageTransforms
except ImportError:
    from training.transforms import ImageTransforms


class FinalModel:
    def __init__(self, model_path, data_path):

        self._path = model_path
        self._destination_path = data_path

        # try to load model on gpus, otherwise use gpus
        try:
            self.model = torch.load(self._path)
        except RuntimeError:
            self.model = torch.load(self._path, map_location=torch.device('cpu'))

        self.val_transforms = None
        self.train_transforms = None
        self._class_names = None
        self._current_prediction = None

        self._set_transforms()

    def sample_random_image(self):
        """
        Samples random image from the validation data by sampling the paths

        :return: path to random sampled image
        :rtype: str
        """

        base_dir = "test"
        base_path = os.path.join(self._destination_path, base_dir)

        self._class_names = os.listdir(base_path)
        crash_or_normal = os.listdir(base_path)[np.random.choice(len(os.listdir(base_path)))]
        intermediate_path = os.path.join(base_path, crash_or_normal)

        image_file = os.listdir(intermediate_path)[np.random.choice(len(os.listdir(intermediate_path)))]
        final_image_file = os.path.join(intermediate_path, image_file)

        return final_image_file

    def _set_transforms(self):
        """
        Sets up the correct transforms specified by the used model

        :return: None
        """
        transforms = ImageTransforms(self._destination_path)
        if "efficientnet" in self._path.lower():
            self.train_transforms = transforms.efficient_net_train_transforms
            self.val_transforms = transforms.efficient_net_val_transforms

        elif "vgg19" in self._path.lower():
            self.train_transforms = transforms.vgg19_train_transforms
            self.val_transforms = transforms.vgg19_val_transforms

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
        return final_image

    def predict_raw_image(self, image):
        """
        Predicts the class of a raw image

        :param image:
        :return:
        """
        transformed_image = self.preprocess_image(image)
        out = self.model(transformed_image)
        out = torch.max(out, dim=1)[1].item()
        return out

    def get_class_activation_map(self, input):
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

    def plot_cam(self):
        """
        Plots the class activation map for the image

        :return: NOne
        """
        # get image path
        img_path = self.sample_random_image()
        img_transformed = f.preprocess_image(img_path)

        cam = f.get_class_activation_map(img_transformed)

        # create plots
        fig, (ax1,ax2) = plt.subplots(1, 2)

        # remove fourth dimension and resample so matplotlib can display the image
        ax1.imshow(img_transformed.squeeze().permute(1,2,0))
        ax2.imshow(img_transformed.squeeze().permute(1, 2, 0), alpha=0.9)
        ax2.imshow(cam, alpha=0.5, cmap="jet")
        ax1.set_title(self._current_prediction)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    with open(r"training/config.yml") as f:
        config = yaml.safe_load(f)

        model_path = config["specific_model_name_to_use"]
        data_path = config["create_train_test_dir"]["destination_path"]

    f = FinalModel(model_path,data_path)

    while True:
        try:
            f.plot_cam()
        except RuntimeError:
            continue