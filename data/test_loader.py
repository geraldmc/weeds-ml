import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from data.transforms import data_transforms

class DeepWeeds_Test(Dataset):

    def __init__(self, csv_file):
        """
        """
        self.root = 'data/test/'
        self.csv_file = csv_file
        self.transform = data_transforms['default']
        
        self.csv_data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.csv_data.Filename[idx])
        label = self.csv_data.Label[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label