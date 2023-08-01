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

image_size=48
margin=0
mtcnn = MTCNN(image_size=image_size, margin=margin)

resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
num_features = resnet.last_linear.in_features
resnet.last_linear = nn.Linear(num_features, 7)

transform = transforms.ToTensor()

dataset_path = 'train' 

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
        img = Image.open(img_path)
        img_resized = img.resize((224, 224))
        img_color = ImageOps.colorize(img_resized, "black", "white")
        img_cropped = mtcnn(img_color)
        if self.transform:
            img_tensor = self.transform(img_color)
        return img_tensor

train_dataset = FaceDataset(dataset_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
