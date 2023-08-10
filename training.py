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

resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
num_features = resnet.logits.in_features
resnet.logits = nn.Linear(num_features, 7)

def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet.to(device)

    # Number of epochs
    num_epochs = 6
   # Training loop
    for epoch in range(num_epochs):
        resnet.train()  # Set the model to training mode
        train_running_loss = 0.0
        train_correct_preds = 0  # Track number of correct predictions
        train_total_preds = 0  # Track total number of predictions

        print("New Epoch")
        x = 0
        for images, labels in pp.train_loader:
            images, labels = images.to(device), labels.to(device)
            print(x)
            x=x+1
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            
            print(loss)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            train_total_preds += labels.size(0)
            train_correct_preds += (predicted == labels).sum().item()
            print((predicted == labels).sum().item())
        
        # Print epoch loss
        train_epoch_loss = train_running_loss / len(pp.train_dataset)
        train_epoch_accuracy = train_correct_preds / train_total_preds
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.4f}')
    
def worker():
    train_model()
if __name__=='__main__':
    p = Process(target=worker)
    p.start()
    p.join()

torch.save(resnet, 'trained_model.pt')
