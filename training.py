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
import preprocessing as pp
from multiprocessing import Process

def train_model():
    print('yes')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pp.resnet.parameters(), lr=0.001)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pp.resnet.to(device)

    # Number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        pp.resnet.train()  # Set the model to training mode
        running_loss = 0.0
        print(0)
        for images, labels in pp.train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            print(1)
            # Forward pass
            outputs = pp.resnet(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Print epoch loss
        epoch_loss = running_loss / len(pp.train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
def worker():
    train_model()
if __name__=='__main__':
    p = Process(target=worker)
    p.start()
    p.join()
