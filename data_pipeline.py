import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
import config
from collections import Counter


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
            transforms.RandomRotation(degrees=10),

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

        # it is time to identify the number of evaluation and train dataset elements
        val_size = int(validation_split * len(train_set))
        train_size = len(train_set) - val_size

        # Fix random seed. Every run produces the SAME train/validation split.
        generator = torch.Generator().manual_seed(seed)

        # I am splitting the dataset into two parts
        train_indices, val_indices = random_split(
            train_set,
            [train_size, val_size],
            generator=generator
        )

        # splitting the train dataset
        train_dataset = Subset(
            train_set,
            indices=train_indices.indices
        )

        # splitting the evaluation dataset
        val_dataset = Subset(
            eval_set,
            indices=val_indices.indices
        )

        # load test dataset
        test_dataset = datasets.ImageFolder(
            root=config.test_data,
            transform=self.eval_transform
        )

        # making tensor from loaded train images
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

        # making tensor from validation images
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        # making tensor from test images
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        return (
            train_loader,
            val_loader,
            test_loader,
            train_set.classes
        )

    def _print_class_distribution(self, dataset):
        counts = Counter(dataset.targets)

        print("\nClass Distribution\n")

        for idx, class_name in enumerate(dataset.classes):
            print(

                f"{class_name}: {counts[idx]}"

            )