import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import os
import shutil


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ImageStats:
    def __init__(self, image_path):
        self._image_path = image_path
        self.image_data = None
        self._load_images()

    def _load_images(self):

        transforms = tt.Compose([tt.Resize((200, 200)),
                                 tt.ToTensor()])

        images = ImageFolder(self._image_path, transforms)

        self.image_data = DataLoader(images, batch_size=16)

    def compute_stats(self):
        all_means = torch.zeros((len(self.image_data), 3))
        all_stds = torch.zeros((len(self.image_data), 3))
        for num, batch in enumerate(self.image_data):
            data, label = batch
            for channel in range(3):
                all_means[num][channel] = torch.mean(data[:, channel, :, :]).item()
                all_stds[num][channel] = torch.std(data[:, channel, :, :]).item()

        return torch.mean(all_means, dim=0), torch.mean(all_stds, dim=0)


def create_train_and_test_dir(img_data_path, split_ratio, destination):
    all_img_dict = {"train": {}, "test": {}}

    if os.path.exists(destination):
        print("removed old folder")
        shutil.rmtree(destination)

    for img_class in os.listdir(img_data_path):
        length_folder = len(os.listdir(os.path.join(img_data_path, img_class)))
        all_imgs = np.zeros(shape=(length_folder,), dtype="object")
        for index, file in enumerate(os.listdir(os.path.join(img_data_path, img_class))):
            all_imgs[index - 1] = (os.path.join(img_data_path,img_class, file))

        split_num = int(np.round(split_ratio * length_folder, 0))
        train_imgs = np.random.choice(all_imgs, split_num, replace=False)

        set_of_test_img = set(all_imgs).difference(train_imgs)
        test_imgs = (list(set_of_test_img))
        all_img_dict["train"][img_class] = train_imgs
        all_img_dict["test"][img_class] = test_imgs

    for train_or_test in all_img_dict.keys():
        if str(train_or_test) == "train":
            label = "train"
        else:
            label = "test"
        dest_path = os.path.join(destination, label)
        for key in all_img_dict[label].keys():
            class_name = str(key).replace(" ", "_")
            if not os.path.exists(os.path.join(dest_path, class_name)):
                os.makedirs(os.path.join(dest_path, class_name))
            for path in all_img_dict[label][key]:
                shutil.copyfile(path, os.path.join(dest_path, class_name, os.path.basename(path)))


