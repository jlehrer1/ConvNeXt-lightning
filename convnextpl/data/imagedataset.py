from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd 
import os 

class ImageSet(Dataset):
    def __init__(
        self, 
        image_path, 
        label_path, 
        label_col='labels',
        name_col='file',
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.label_df = pd.read_csv(label_path)

        self.labels = self.label_df.loc[:, label_col]
        self.filenames = self.label_df.loc[:, name_col]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_name, label = self.filenames.iloc[idx], self.labels.iloc[idx]
        img = Image.open(os.path.join(self.image_path, img_name))
        return self.tensor(img), label
