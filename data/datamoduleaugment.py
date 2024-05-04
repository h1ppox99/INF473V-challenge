import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import random
from torchvision.datasets.folder import default_loader


class DuplicatedDataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None, num_copies=5):
        super(DuplicatedDataset, self).__init__(root, transform, target_transform, loader, is_valid_file)
        self.num_copies = num_copies

    def __getitem__(self, index):
        # Adjust the index to account for the number of copies
        adjusted_index = index // self.num_copies
        return super(DuplicatedDataset, self).__getitem__(adjusted_index)


    def __len__(self):
        return super(DuplicatedDataset, self).__len__() * self.num_copies


class DataModuleAugment:
    def __init__(
        self,
        train_dataset_path,
        real_images_val_path,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
    ):
        self.dataset = ImageFolder(train_dataset_path)
        # creating copies of the dataset
        self.dataset = DuplicatedDataset(train_dataset_path, train_transform, num_copies=5)

        # Assign the transform for training dataset
        self.val_transform = val_transform

        # Splitting the dataset into training and validation
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        # Assign the transform for validation dataset
        self.val_dataset.dataset.transform = val_transform

        self.real_images_val_dataset = datasets.ImageFolder(
            real_images_val_path, transform=val_transform
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return {
            "synthetic_val": DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            "real_val": DataLoader(
                self.real_images_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        }
    
      # def custom_collate_fn(self, batch):
    #     new_batch = []
    #     for images, labels in batch:
    #         for _ in range(5):  # Create 5 versions of each image
    #             # Apply three random transformations
    #             transform = transforms.Compose(random.sample(self.augmentation_transforms, 3))
    #             images_aug = transform(images)
    #             new_batch.append((images_aug, labels))
    #     return torch.utils.data.dataloader.default_collate(new_batch)

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #         collate_fn=self.custom_collate_fn  # Use the custom collate function
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.real_images_val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers
    #     )

