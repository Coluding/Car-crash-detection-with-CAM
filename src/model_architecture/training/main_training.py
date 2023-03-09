import sys
import os
import argparse
from azureml.core import Workspace
from vgg19 import VGG19
from efficient_net import EfficientNet


def init_model_with_azure_remote_paths():
    parser = argparse.ArgumentParser()
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
    workspace = Workspace(subscription_id='41389919-46b1-46b4-819a-f1ccb00cc40c',
                        resource_group='gpu_training',
                        workspace_name='ImageClassification',
                        _location="westeu",
                        )
    model = init_model_with_azure_remote_paths()
    model.fit(patience=5, factor=0.1, azure_workspace=workspace) # Learning rate scheduler kwargs


if __name__ == "__main__":
    main()