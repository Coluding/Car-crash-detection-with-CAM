import sys
import os
import argparse
from azureml.core import Workspace
from vgg19 import VGG19
from efficient_net import EfficientNet


def init_model_with_azure_remote_paths():
    parser = argparse.ArgumentParser()
    #os.environ['CUDA_ALLOC_CONF'] = "max_split_size_mb:32"
    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='data',
                        help='data folder mounting point')

    args = parser.parse_args()

    # set data paths
    train_folder = os.path.join(args.data_path, 'train')
    val_folder = os.path.join(args.data_path, 'test')

    model = EfficientNet(remote_run=True, train_path=train_folder, val_path=val_folder)

    return model


def init_model_with_local_paths():
    model = EfficientNet()

    return model


def main():

    ws = Workspace(
        subscription_id="a0375d6b-a4f6-4c9e-9054-e9982f6f6765",
        resource_group="GPU_Test",
        workspace_name="InsuranceImageRecognition",
        _location="westeu"
    )

    model = init_model_with_azure_remote_paths()

    model.fit(azure_workspace=ws, patience=5, factor=0.1) # Learning rate scheduler kwargs
    #model.torch_save_model()
    #model.save_model()


if __name__ == "__main__":

    main()