# File        : dataloader.py
# Authors     : Rohan Karamel
# Date        : Feburary 19th, 2025
# Description : This file contains the custom dataset and dataloader for the UA-DETRAC object detection dataset.

# Built-in libraries
import os

# Third-party libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UADETRACDataset( # Custom dataset for the UA-DETRAC object detection dataset.
            Dataset    # Extends the PyTorch Dataset class
        ):

    def __init__(                        # Initializes the dataset with image and label directories.
            self       : object,         # Instance of the class
            images_dir : str,            # Path to the folder containing image files
            labels_dir : str,            # Path to the folder containing label (txt) files
            transform  : callable = None # Optional transform to be applied on an image
        )   ->           None:           # Void return type

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform  = transform

        # Create a sorted list of image files (assumes .jpg extension)
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(          # Gets the number of images in the dataset.
            self : object # Instance of the class
        )   ->     int:   # The number of images in the dataset

        return len(self.image_files)

    def __getitem__(       # Gets an image and its bounding boxes from the dataset.
            self : object, # Instance of the class
            idx  : int     # Index of the image to retrieve
        )   ->     dict:   # A dictionary containing the image, bounding boxes, and image filename

        # Get image path and corresponding label path
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, os.path.splitext(image_filename)[0] + ".txt")

        # Load image and apply transformation to RGB format
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Read bounding boxes from the label file
        bounding_boxes = []
        try:
            with open(label_path, "r", encoding="utf-8") as label_file: # Open the label file with utf-8 encoding
                for line in label_file: # Each line contains the tuple: (class, x, y, width, height)
                    parts = line.strip().split()

                    if len(parts) != 5: # Check if the label format is valid
                        return ValueError(f"Invalid label format in {label_path}")

                    # Only take the coordinates (x, y, width, height) and convert to float
                    x, y, width, height = map(float, parts[1:])
                    bounding_boxes.append([x, y, width, height])
        except FileNotFoundError: # If the corresponding label file is not found
            bounding_boxes = []   # Set bounding boxes to an empty list

        # Convert bounding boxes to a tensor; if no boxes, create an empty tensor with shape [0, 4]
        bounding_boxes = torch.tensor(bounding_boxes) if bounding_boxes else torch.zeros((0, 4))

        sample = {
            "image": image,
            "bounding_boxes": bounding_boxes, 
            "image_filename": image_filename
        }

        return sample

def custom_collate(   # Custom collate function to handle variable-size bounding box tensors.
        batch : list  # List of samples from the dataset
    )   ->      dict: # A dictionary containing the batched image, bounding boxes, and image filenames

    # Stack images 
    images = torch.stack([sample["image"] for sample in batch], dim=0)

    # Keep bounding boxes as a list since they have different sizes
    bounding_boxes = [sample["bounding_boxes"] for sample in batch]

    # Collect filenames into a list
    image_filenames = [sample["image_filename"] for sample in batch]

    return {
        "image"          : images,
        "bounding_boxes" : bounding_boxes,
        "image_filename" : image_filenames
    }

def get_dataloader(                  # Creates a DataLoader for the UA-DETRAC dataset.
        images_folder : str,         # Path to folder containing images
        labels_folder : str,         # Path to folder containing labels
        batch_size    : int  = 16,   # Number of samples per batch
        shuffle       : bool = True, # Whether to shuffle the dataset
    )   ->              DataLoader:  # Configured PyTorch DataLoader

    # Define the transformation to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
    ])

    # Create the dataset
    dataset = UADETRACDataset(images_folder, labels_folder, transform = transform)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,                      # The dataset to load
        batch_size  = batch_size,     # Number of samples per batch
        shuffle     = shuffle,        # Whether to shuffle the dataset
        collate_fn  = custom_collate, # Custom collate function to handle variable-size bounding boxes
        num_workers = os.cpu_count(), # Use as many workers as there are cores
        pin_memory  = True,           # Enable faster host-to-device transfer
    )

    # Return the DataLoader
    return dataloader

def main(): # Example usage
    train_images_dir = "./data/images/train" # Path to the training images
    train_labels_dir = "./data/labels/train" # Path to the training labels
    val_images_dir   = "./data/images/val"   # Path to the validation images
    val_labels_dir   = "./data/labels/val"   # Path to the validation labels

    # Create dataloaders for training and validation
    train_loader = get_dataloader(train_images_dir, train_labels_dir, batch_size=8, shuffle=True)
    val_loader   = get_dataloader(val_images_dir,   val_labels_dir,   batch_size=8, shuffle=False)

    print("\nLoading Training Data...")

    # Print out one batch of training data
    for example_batch in train_loader:
        example_images = example_batch["image"]
        bboxes         = example_batch["bounding_boxes"]
        print(f"Batch size         : {example_images.size(0)}")
        print(f"Image tensor shape : {example_images.shape}")
        print(f"Bounding boxes     : \n{bboxes}")
        break # Break after the first batch

    print("Training Data Loaded." + "\n\n" + "Loading Validation Data...")

    # Print out one batch of validation data
    for example_batch in val_loader:
        example_images = example_batch["image"]
        bboxes         = example_batch["bounding_boxes"]
        print(f"Batch size         : {example_images.size(0)}")
        print(f"Image tensor shape : {example_images.shape}")
        print(f"Bounding boxes     : \n{bboxes}")
        break # Break after the first batch

    print("Validation Data Loaded.")

if __name__ == "__main__":
    main()
