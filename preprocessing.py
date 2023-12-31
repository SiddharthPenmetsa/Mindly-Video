import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.optim as optim

transform = transforms.ToTensor()

dataset_path0 = 'train' 
dataset_path1 = 'test'

class FaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.file_list = []
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            for image in os.listdir(folder_path):
                if image.endswith(".png") or image.endswith(".jpg"):
                    self.file_list.append(os.path.join(folder_path, image))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        folder_path = os.path.dirname(img_path)
        if folder_path == 'train\\angry' or folder_path == 'test\\angry':
            label = 0
        if folder_path == 'train\\disgust' or folder_path == 'test\\angry':
            label = 1
        if folder_path == 'train\\fear' or folder_path == 'test\\angry':
            label = 2
        if folder_path == 'train\\happy' or folder_path == 'test\\angry':
            label = 3
        if folder_path == 'train\\neutral' or folder_path == 'test\\angry':
            label = 4
        if folder_path == 'train\\sad' or folder_path == 'test\\angry':
            label = 5
        if folder_path == 'train\\surprise' or folder_path == 'test\\angry':
            label = 6
        img = Image.open(img_path)
        img_resized = img.resize((224, 224))
        img_color = ImageOps.colorize(img_resized, "black", "white")
        if self.transform:
            img_tensor = self.transform(img_color)
        tensor1 = img_tensor.cpu()
        img_tensor = tensor1.numpy()
        return img_tensor, label

train_dataset = FaceDataset(dataset_path0, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)

test_dataset = FaceDataset(dataset_path1, transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)
