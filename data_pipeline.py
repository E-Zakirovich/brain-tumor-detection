import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import config


class DataPipeline:

    def __init__(self):

        self.train_transform = transforms.Compose([
            # changing the size of the images. The size of some images might not be same as others, so I need all images must be in same size
            transforms.Resize(
                (config.img_size, config.img_size)
            ),

            # to avoid overfitting, I want to flip half of train dataset
            transforms.RandomHorizontalFlip(p=0.5),

            # to avoid overfitting, I want to rotate images between -10 and 10 degree randomly
            transforms.RandomResizedCrop(degrees=10),

            # to learn my cnn, my dataset must be in tensors format not like a image or sth
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        ])