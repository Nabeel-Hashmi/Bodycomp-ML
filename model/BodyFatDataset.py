import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class BodyFatDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        self.label_map = {
            "0-5": 0,
            "5-10": 1,
            "10-15": 2,
            "15-20": 3,
            "20-25": 4,
            "25-30": 5,
            "30+": 6
        }

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.df["class_id"] = self.df["label_bucket"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        label = int(row["class_id"])

        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
    
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])