import torch
import torchvision.transforms as tt
import os

try:
    from .utils import ImageStats
except ImportError:
    from utils import ImageStats


class ImageTransforms:
    def __init__(self, destination_path):
        image_stats_train = ImageStats(destination_path)
        try:
            self.stats = image_stats_train.load_stats_config()
            print("using config")
        except Exception as E:
            print(E)
            self.stats = image_stats_train.compute_stats()
            image_stats_train.save_stats_config()

        self.efficient_net_val_transforms = tt.Compose([
            tt.Resize((256, 256), interpolation=tt.InterpolationMode.BILINEAR),
            tt.ToTensor(),
            tt.Normalize(*self.stats, inplace=True)
        ])

        self.efficient_net_train_transforms = tt.Compose([
            tt.Resize((256, 256), interpolation=tt.InterpolationMode.BILINEAR),
            tt.RandomCrop((240, 240)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(),
            tt.ToTensor(),
            tt.Normalize(*self.stats, inplace=True)
        ])

        self.vgg19_val_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.ToTensor(),
            tt.Normalize(*self.stats, inplace=True)
        ])

        self.vgg19_train_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.RandomCrop((224, 224)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(),
            tt.ToTensor(),
            tt.Normalize(*self.stats, inplace=True)
        ])

    def denormalize_img(self, img, stats):
        means, stds = stats
        return img * torch.tensor(stds).view(3,1,1) + torch.tensor(means).view(3,1,1)
