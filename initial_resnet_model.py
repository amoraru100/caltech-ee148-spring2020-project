from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from skimage import io, transform
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings("ignore")


'''
Create a dataloader
'''
class CarsDataset(Dataset):

    def __init__(self, annos_path, data_dir, transform=None):
        """
        Args:
            annos_path (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.car_details = pd.read_csv(annos_path)
        self.car_details = np.array(self.car_details)

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_details)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_details[idx][0])
        image = Image.open(img_name)
        num_channel = len(image.split())
        car_class = self.car_details[idx][5]
        car_details = self.car_details[idx][6]
        x1, y1, x2, y2 = self.car_details[idx][1], self.car_details[idx][2], self.car_details[idx][3], self.car_details[idx][4]
        bounding_box = [x1, y1, x2, y2]

        if self.transform:
            image = self.transform(image)
     
        return image, bounding_box, car_class, car_details


'''
Split the training dataset into training and validation sets
'''
def train_valid_split(train_dataset):
    np.random.random(2020)
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    
    subset_indices_train = indices[: int(0.85*len(train_dataset))]
    subset_indices_valid = indices[int(0.85*len(train_dataset)) :]

    assert (len(subset_indices_train) + len(subset_indices_valid) == len(train_dataset))
    
    np.save("train_indices.npy", subset_indices_train)
    np.save("valid_indices.npy", subset_indices_valid)
    
    return subset_indices_train, subset_indices_valid


'''
Function for training the model
'''
def train_model(model, criterion, optimizer, scheduler, n_epochs):
    model.train()
    all_losses = []
    all_accuracies = []
    for epoch in range(n_epochs):
        losses = []
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, bbox, labels, details = data
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            correct += (labels==predicted).sum().item() 
        accuracy = 100/64*correct/len(train_loader)
        print("Epoch %s, Training Accuracy: %.4f %%, Training Loss: %.4f" % (epoch+1, accuracy, np.average(losses)))
        all_losses.append(np.average(losses))
        all_accuracies.append(accuracy)
    return all_losses, all_accuracies
    
'''
Function for evaluating the model on the validation set
'''    
def test_model(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, bbox, labels, details = data
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    print('Validation accuracy: %.4f %%' % (test_acc))
    return test_acc

'''
Train a simple ResNet model and store the training losses and accuracies
'''
def main():
    cars_data_train = CarsDataset('train_annos_cleaned.csv', train_path, transform=transforms.Compose(
        [transforms.Resize(100), transforms.RandomSizedCrop(100), transforms.ToTensor()]))

    print("Loading training and validation sets")
    
    subset_indices_train = np.load("train_indices.npy")
    subset_indices_valid = np.load("valid_indices.npy")

    train_loader = torch.utils.data.DataLoader(cars_data_train, batch_size=64, sampler=SubsetRandomSampler(subset_indices_train))
    val_loader = torch.utils.data.DataLoader(cars_data_train, batch_size=64, sampler=SubsetRandomSampler(subset_indices_valid))

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 196)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    print ("Training model...")
    
    training_losses, training_accuracies = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=10)
    test_accuracy = test_model(model_ft, val_loader)
    
    print("Saving initial ResNet model...")
    
    torch.save(model_ft.state_dict(), "updated_resnet.pt")

     print ("Done!")
        
if __name__ == "__main__":
    main()