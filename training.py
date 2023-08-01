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
    pp.resnet.train()  

    num_epochs = 10  

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pp.resnet.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(pp.train_loader, 0):
            
            inputs = data
            print(data.shape)
            
            optimizer.zero_grad()

            outputs = pp.resnet(inputs)
            loss = criterion(outputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    
def worker():
    train_model()
if __name__=='__main__':
    p = Process(target=worker)
    p.start()
    p.join()
