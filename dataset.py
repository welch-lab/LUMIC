import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import numpy as np

class remove_channel(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() <= self.p:
            img[0] = torch.zeros(1, *img.shape[1:])
        if np.random.rand() <= self.p:
            img[1] = torch.zeros(1, *img.shape[1:])
        if np.random.rand() <= self.p:
            img[2] = torch.zeros(1, *img.shape[1:])
        return img

class DiffusionDatasetSuperRes(Dataset):
    def __init__(self, df):
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256)])
        self.additional_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), remove_channel(0.2)])
        self.dino_transforms = torchvision.transforms.Compose([ torchvision.transforms.Pad(50)])
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.df.iloc[idx, 0])
        counter = 0
        img = self.transforms(image)
        while(torch.count_nonzero(torchvision.transforms.ToTensor()(img)) / (256 * 256 * 3) < 0.3 and counter < 50):
            img = self.transforms(image)
            counter += 1
        img = self.additional_transforms(img)
        return img, self.dino_transforms(img)

class DiffusionDataset(Dataset):
    def __init__(self, df):
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256)])
        self.additional_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), remove_channel(0.2)])
        self.img_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(64)])
        self.dino_transforms = torchvision.transforms.Compose([ torchvision.transforms.Pad(50)])
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.df.iloc[idx, 0])
        counter = 0
        img = self.transforms(image)
        while(torch.count_nonzero(torchvision.transforms.ToTensor()(img)) / (256 * 256 * 3) < 0.3 and counter < 50):
            img = self.transforms(image)
            counter += 1
        img = self.additional_transforms(img)
        return self.img_transforms(img), self.dino_transforms(img)

class DiffusionDataset1D(Dataset):
    def __init__(self, df, smile_emb):
        self.df = df
        self.smiles = smile_emb
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
        self.pad = torchvision.transforms.Compose([torchvision.transforms.Pad(50)])
        self.control = df
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        counter = 0
        counter1 = 0
        image = Image.open(self.df.iloc[idx, 0])
        img = self.transforms(image)
        smile = self.df.iloc[idx, 2]
        smile_vec = self.smiles[self.smiles["smiles"] == smile].iloc[0, :256]
        smile_vec = torch.tensor([smile_vec], dtype = torch.float32)
        cell_type = self.df.iloc[idx, 3]
        
        subset = self.control[self.control["CellLine"] == cell_type]
        subset = subset[subset["smile"] == "CS(=O)C"]
        control = subset.sample(n = 1)
        
        control_img = Image.open(control.iloc[0, 0])
        control_img = self.transforms(control_img)
        while(torch.count_nonzero(control_img)/ (256 * 256 * 3) < 0.3 and counter1 < 50):
            control = subset.sample(n = 1)
            control_img = Image.open(control.iloc[0, 0])
            control_img = self.transforms(control_img)
            counter1 += 1

        while(torch.count_nonzero(img) / (256 * 256 * 3) < 0.3 and counter < 50):
            img = self.transforms(image)
            counter += 1
            
        img = self.pad(img)
        control_img = self.pad(img)
        return img, smile_vec, control_img

class DiffusionDataset1DSample(Dataset):
    def __init__(self, df, smile_emb):
        self.df = df
        self.smiles = smile_emb
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor()])
        self.pad = torchvision.transforms.Compose([torchvision.transforms.Pad(50)])
        self.control = df
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        counter = 0
        counter1 = 0
        image = Image.open(self.df.iloc[idx, 0])
        img = self.transforms(image)
        smile = self.df.iloc[idx, 2]
        smile_vec = self.smiles[self.smiles["smiles"] == smile].iloc[0, :256]
        smile_vec = torch.tensor([smile_vec], dtype = torch.float32)
        cell_type = self.df.iloc[idx, 3]
        subset = self.control[self.control["CellLine"] == cell_type]
        subset = subset[subset["smile"] == "CS(=O)C"]
        control = subset.sample(n = 1)
        control_img = Image.open(control.iloc[0, 0])
        control_img = self.transforms(control_img)
        while(torch.count_nonzero(control_img)/ (256 * 256 * 3) < 0.3 and counter1 < 50):
            control = subset.sample(n = 1)
            control_img = Image.open(control.iloc[0, 0])
            control_img = self.transforms(control_img)
            counter1 += 1
        
        while(torch.count_nonzero(img) / (256 * 256 * 3) < 0.3 and counter < 50):
            img = self.transforms(image)
            counter += 1
        
        unpadded_img = img
        small_img = torchvision.transforms.Resize(64)(img)
        img = self.pad(img)
       
        control_img = self.pad(img)
        return img, smile_vec, control_img, unpadded_img, small_img