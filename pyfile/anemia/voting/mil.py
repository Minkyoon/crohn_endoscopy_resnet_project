import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from collections import defaultdict
from torchvision import datasets, transforms

class BagDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform

        # Read the csv file
        self.data_info = pd.read_csv(csv_file)

        # Group the image_paths by accession_number to form bags
        self.bags = defaultdict(list)
        for _, row in self.data_info.iterrows():
            self.bags[row['accession_number']].append((row['filepath'], row['label']))

        # Convert the defaultdict to list for indexing
        self.bags = list(self.bags.items())

    def __getitem__(self, index):
        bag_images = []
        bag_label = None
        for image_path, label in self.bags[index][1]:
            # Load the npy file
            image = np.load(image_path)
            if self.transform:
                image = self.transform(image)
            bag_images.append(image)
            if bag_label is None:
                bag_label = label  # Assuming all images in a bag have the same label

        bag_images = torch.stack(bag_images)
        return bag_images, bag_label

    def __len__(self):
        return len(self.bags)
    
transform = transforms.Compose([
    transforms.ToTensor(),       
])


# Initialize the dataset
bag_dataset = BagDataset(csv_file="/home/minkyoon/crom/pyfile/anemia/voting/train.csv", transform=transform)

# Initialize the dataloader
dataloader = DataLoader(bag_dataset, batch_size=32, shuffle=True)

# Loop over the data
for images, labels in dataloader:
    # Do something with the images and labels...
    pass
