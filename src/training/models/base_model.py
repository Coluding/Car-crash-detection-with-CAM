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
from src.training.models.utils import EarlyStopper, ImageStats
import os


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        with open("../../config.yml") as y:
            self._config_file = yaml.safe_load(y)

        self.today = datetime.datetime.today().strftime("Y-m-d")
        self.tb = SummaryWriter()
        self._specific_config_file = None
        self.train_transforms = None
        self.val_transforms = None
        self.model = None
        self.classes = None
        self.train_loader = None
        self.val_loader = None
        self.history = []

    @staticmethod
    def accuracy(output, labels):
        _, preds = torch.max(output, dim=1)
        return torch.tensor(torch.sum(preds == labels) / len(preds))

    @abstractmethod
    def _init_transforms(self):
        pass

    @abstractmethod
    def _init_backbone_model(self, new_model=True):
        pass

    @abstractmethod
    def _add_classifier(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def _set_up_model(self):
        """
        Initializes the whole model. It is called in the constructor of the child class

        :return: None
        """
        self._init_backbone_model()
        self._add_classifier()

    def _collect_hyperparams(self):
        """
        Collects hyperparams from config file

        :return: None
        """
        self.epochs = self._specific_config_file["epochs"]
        self.batch_size = self._specific_config_file["batch_size"]
        self.learning_rate = self._specific_config_file["learning_rate"]
        self.train_backbone_weights = self._specific_config_file["train_backbone_params"]
        self.train_test_split_ratio = self._specific_config_file["train_test_split_ratio"]
        self.classifier_layer = self._specific_config_file["classifier_layer"]
        self.activation_function = self._specific_config_file["activation_function"]

        self.hparams_dict = {"epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate,
                             "train_backbone_weights": self.train_backbone_weights,
                             "tran_test_split_ratio": self.train_test_split_ratio,
                             "activation_function": self.activation_function}

    def _get_activation_function(self):
        """
        Assigns the activation function specified in the config file

        :return: pytorch activation function used by the _create_layers() method
        :rtype: torch.nn.modules.activation
        """

        if self.activation_function == "tanh":
            return nn.Tanh()
        elif self.activation_function == "relu":
            return nn.ReLU()
        elif self.activation_function == "sigmoid":
            return nn.Sigmoid()
        elif self.activation_function == "elu":
            return nn.ELU()
        elif self.activation_function == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError("Please make sure to enter one of the following activation fucntions in the config file:"
                             "'tanh', 'relu', 'sigmoid', 'elu' or 'leakyrelu'!")

    def _create_layers(self, config_list_of_layers):
        final_layer_list = []
        layer_dict = {}
        activation_function = self._get_activation_function()
        for index, layer in enumerate(config_list_of_layers):
            in_shape = list(layer.values())[0][0]
            out_shape = list(layer.values())[0][1]
            dropout_rate = list(layer.values())[0][2]

            nn_layer = nn.Linear(in_features=in_shape, out_features=out_shape)
            final_layer_list.append(nn_layer)
            layer_dict[f"layer_{index}_shape_in"] = in_shape
            layer_dict[f"layer_{index}_shape_out"] = out_shape

            if dropout_rate != 0:
                dropout_layer = nn.Dropout(dropout_rate)
                final_layer_list.append(dropout_layer)
                layer_dict[f"layer_{index}_dropoutRrate"] = dropout_rate

            if index != len(config_list_of_layers):
                final_layer_list.append(activation_function) # TODO tanh maybe

        final_layer_list.append(nn.LogSoftmax(dim=1))
        return final_layer_list, layer_dict

    def evaluate(self):
        with torch.no_grad():
            self.eval()
            outputs = [self.validation_step(batch) for batch in self.val_loader]
            return self.validation_epoch_end(outputs)

    def _load_images(self):
        train_path = os.path.join(self._config_file["image_path"], "train")
        val_path = os.path.join(self._config_file["image_path"], "test")
        train_images = ImageFolder(train_path, self.train_transforms)
        val_images = ImageFolder(val_path, self.val_transforms)
        self.classes = train_images.classes
        # more augmenting !!

        train_loader = DataLoader(train_images, batch_size=self._specific_config_file["batch_size"], num_workers=2,
                                  shuffle=True)
        val_loader = DataLoader(val_images, batch_size=self._specific_config_file["batch_size"], num_workers=2,
                                shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _get_class_weights(self):
        image_number_sum = 0
        total_image_numbers = []
        path = self._config_file["image_path"]
        for image_class in os.listdir(path):
            number_images = len(os.listdir(os.path.join(path, image_class)))
            image_number_sum += number_images
            total_image_numbers.append(number_images)

        total_image_numbers = torch.tensor(total_image_numbers, requires_grad=False)
        class_weights = total_image_numbers / image_number_sum

        return class_weights

    def forward(self, xb):
        out = self.model(xb)
        return out

    def training_step(self, batch, use_class_weights=True):
        images, labels = batch
        out = self(images)
        if use_class_weights:
            weights = self._get_class_weights()
            loss = F.nll_loss(out, labels, weight=weights)
        else:
            loss = F.F.nll_loss(out, labels)
        acc = self.accuracy(out, labels)
        print(f"train step loss: {loss}")

        return loss, acc

    def validation_step(self, batch):
        self.eval()
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        accuracy = self.accuracy(out, labels)
        print(f"val step loss: {loss}")
        print(f"Validation predictions: {torch.max(out, dim=1)[1]}")
        print(f"True targets: {labels}")
        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_losses = torch.stack(batch_losses).mean().item()
        epoch_acc = torch.stack(batch_acc).mean().item()
        return {"val_loss": epoch_losses, "val_acc": epoch_acc}

    def _epoch_end_val(self, epoch, results):
        """
        Prints validation epoch summary after every epoch

        :param epoch: epoch number
        :type epoch: int
        :param results: results from the evaluate method
        :type results: dictionary
        :return: None
        """
        print(f"Epoch[{epoch}]: val_loss: {results['val_loss']}"
              f" val_acc: {results['val_acc']}")

    def fit(self, optim=torch.optim.Adam, lrs=torch.optim.lr_scheduler.ReduceLROnPlateau, use_class_weights=True,
            **kwargs):
        """
        Fits the model with specified hyperparams of the config file. If a learning rate scheduler is used, the **kwargs
        can be used to give the scheduler arguments

        :param use_class_weights: True if class weights should be used for the loss function
        :param lrs: Pytorch learning rate scheduler
        :param optim: Pytorch optimizer
        :param kwargs: additional keyword arguments for the learning rate scheduler
        :return: None
        """
        early_stopping = self._specific_config_file["early_stopping"]
        optimizer = optim(self.parameters(), lr=self.learning_rate)
        lrs = lrs(optimizer, **kwargs)

        if early_stopping:
            early_stopper = EarlyStopper(patience=10, min_delta=0.3)

        max_acc = 0

        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            train_accs = []
            for batch_number, batch in enumerate(self.train_loader):
                optimizer.zero_grad()
                print(f"new batch: {batch_number}")
                loss, train_acc = self.training_step(batch, use_class_weights)
                loss.backward()
                train_losses.append(loss)
                train_accs.append(train_acc)

                optimizer.step()


            print(f"Epoch {epoch + 1} done!")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            result = self.evaluate()

            result["learning_rate"] = optimizer.param_groups[0]["lr"]
            result["train_loss"] = torch.stack(train_losses).mean().item()
            result["train_acc"] = torch.stack(train_accs).mean().item()

            self.tb.add_scalar("Train loss", result["train_loss"], epoch)
            self.tb.add_scalar("Train accuracy", result["train_acc"], epoch)
            self.tb.add_scalar("Val accuracy", result["val_acc"], epoch)
            self.tb.add_scalar("Val loss", result["val_loss"], epoch)
            self.tb.add_scalar("Learning rate", result["learning_rate"], epoch)
            print(f'Epoch train loss: {result["train_loss"]}, Epoch train accuracy: {result["train_acc"]}')
            self._epoch_end_val(epoch + 1, result)
            self.history.append(result)

            # TODO: Fix LRS with threshhold, factor and mode
            # The learning rate is reduced when the val loss is on a plateau, meaning it does not make significant
            # changes --> local minima
            lrs.step(metrics=result["val_loss"])

            if max_acc < result["val_acc"]:
                max_acc = result["val_acc"]

            # If early stopping is set to true in the config file the early stopper checks after every loop for early
            # stop criterion
            if early_stopping:
                if early_stopper.early_stop(result["val_loss"]):
                    print(f"Early stopping in epoch: {epoch}")
                    break
                else:
                    print("No early stopping!")

        self.tb.add_hparams(self.hparams_dict, {"hparam/max_accuracy": max_acc})
        self.tb.close()

    def print_summary(self, input_size):
        print(torchinfo.summary(self.model, input_size=input_size))
