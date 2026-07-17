import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import config


class DataPipeline:

    def __init__(self):
        self.train_transform = transforms.Compose([

            # the size of some images might not be same as others, so I need all images must be in same size
            transforms.Resize(
                (config.img_size, config.img_size)
            ),

            # to avoid overfitting, I want to flip half of train dataset
            transforms.RandomHorizontalFlip(p=0.5),

            # to avoid overfitting, I want to rotate images between -10 and 10 degree randomly
            transforms.RandomResizedCrop(degrees=10),

            # to teach my cnn, my dataset must be in tensors format not like an image or sth
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        ])

        self.eval_transform = transforms.Compose([

            # the size of some images might not be same as others, so I need all images must be in same size
            transforms.Resize(
                (config.img_size, config.img_size)
            ),

            # to teach my cnn, my dataset must be in tensors format not like an image or sth
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_loaders(
            self,
            batch_size: int = config.batch_size,
            validation_split: float | int = 0.15, seed: int = 42
    ):
        # loading the train dataset
        train_set = datasets.ImageFolder(
            root=config.train_data,
            transform=self.train_transform
        )

        # loading the validation dataset
        eval_set = datasets.ImageFolder(
            root=config.train_data,
            transform=self.eval_transform
        )

        