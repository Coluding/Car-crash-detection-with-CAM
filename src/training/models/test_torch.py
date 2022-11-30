import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch.nn as nn
import yaml
import torch.nn.functional as F
import torchvision
import pickle
import datetime
from PIL import Image
import torchinfo
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopper
import os

path = r'C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\data'

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
