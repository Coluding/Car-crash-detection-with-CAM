import torch
import torchvision.transforms as tt
import os

try:
    from .utils import ImageStats
except ImportError:
    from utils import ImageStats


class ImageTransforms:
    def __init__(self, destination_path):
        image_stats_train = ImageStats(os.path.join(destination_path))
        stats = image_stats_train.compute_stats()

        self.efficient_net_val_transforms = tt.Compose([
            tt.Resize((256, 256), interpolation=tt.InterpolationMode.BILINEAR),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

        self.efficient_net_train_transforms = tt.Compose([
            tt.Resize((256, 256), interpolation=tt.InterpolationMode.BILINEAR),
            tt.RandomCrop((240, 240)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

        self.vgg19_val_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])

        self.vgg19_train_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.RandomCrop((224, 224)),
            tt.RandomRotation(30),
            tt.RandomVerticalFlip(),
            tt.RandomHorizontalFlip(),
            tt.ColorJitter(),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True)
        ])


