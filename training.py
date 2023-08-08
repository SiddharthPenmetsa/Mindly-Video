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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pp.resnet.parameters(), lr=0.0001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pp.resnet.to(device)

    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        pp.resnet.train()  # Set the model to training mode
        correct_preds = 0  # Track number of correct predictions
        total_preds = 0  # Track total number of predictions

        print("New Epoch")

        for images, labels in pp.train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = pp.resnet(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        # Print epoch loss
        epoch_loss = running_loss / len(pp.train_dataset)
        epoch_accuracy = correct_preds / total_preds
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
def worker():
    train_model()
if __name__=='__main__':
    p = Process(target=worker)
    p.start()
    p.join()
