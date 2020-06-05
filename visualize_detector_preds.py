from __future__ import print_function, division
import os
import torch
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle
import pandas as pd
import json

# Define the Dataset
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
        file_name = self.car_details[idx][0]
        image = Image.open(img_name)
        num_channel = len(image.split())
        car_class = self.car_details[idx][5]
        car_details = self.car_details[idx][6]
        x1, y1, x2, y2 = self.car_details[idx][1], self.car_details[idx][2], self.car_details[idx][3], self.car_details[idx][4]
        bounding_box = torch.as_tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        area = (bounding_box[:, 3] - bounding_box[:, 1]) * (bounding_box[:, 2] - bounding_box[:, 0])

        # Pytorch's Faster RCNN recognizes labels of 0 as background
        if car_class == 0:
            car_class = 196
        
        target = {}
        target['boxes'] = bounding_box
        target['labels'] = torch.as_tensor([car_class], dtype=torch.int64)
        target['image_id'] = torch.as_tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd = torch.zeros(1, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target, car_details, file_name

# Setup paths
train_path = './data/cars_train/'
test_path = './data/cars_test/'
devkit_path = './data/cars_devkit'

train_annos_path = devkit_path + '/cars_train_annos.csv'
test_annos_path = devkit_path + '/cars_test_annos.csv'
cars_meta_path = devkit_path + '/cars_meta.csv'

# Get training and validation predictions and targets
with open('val_preds.json','r') as f:
    val_preds = json.load(f)
with open('val_targets.json','r') as f:
    val_targets = json.load(f)
with open('train_preds.json','r') as f:
    train_preds = json.load(f)
with open('train_targets.json','r') as f:
    train_targets = json.load(f)

# load in the dataset
cars_data_train = CarsDataset('train_annos_cleaned.csv', train_path)
cars_meta = pd.read_csv(cars_meta_path, header=None)

def show_correct_preds():
    fig, axs = plt.subplots(3, 3, figsize=(15,15)) 
    idx = 0
    for i in range(3):
        for j in range(3):
            image, target, true_label, file_name = cars_data_train[idx]
            true_box = target['boxes'][0]
            true_id = target['labels'][0]
            pred = train_preds[file_name]

            pred_label = cars_meta.loc[train_preds[file_name]['labels'][0], 0]
            pred_box = pred['boxes'][0]

            for k in range(len(pred['labels'])):
                if pred['labels'][k] == true_id:
                    pred_label = true_label
                    pred_box = pred['boxes'][k]
                    break

            if (pred_label == true_label):
                c = 'g'
            else:
                c = 'r'

            xy = pred_box[0], pred_box[1]
            width = pred_box[2] - pred_box[0]
            height = pred_box[3] - pred_box[1]
            rect = Rectangle(xy, width, height, fill=False, color=c, linewidth=2)
            
            axs[i, j].imshow(image)
            axs[i, j].set_title('Actual label: {} \n Predicted Label: {}'.format(true_label, pred_label), fontdict = {'color': c})
            axs[i, j].add_patch(rect)
            idx += 1   

def show_preds(score_thresh = 0):
    fig, axs = plt.subplots(2, 2, figsize=(15,15)) 
    idx = 0
    for i in range(2):
        for j in range(2):
            image, target, true_label, file_name = cars_data_train[idx+3]
            true_box = target['boxes'][0]
            true_id = target['labels'][0]
            pred = train_preds[file_name]

            axs[i, j].imshow(image)

            for k in range(len(pred['labels'])):
                if pred['scores'][k]>score_thresh:
                    pred_box = pred['boxes'][k]

                    if pred['labels'][k] == true_id:
                        c = 'g'
                    else:
                        c = 'r'

                    xy = pred_box[0], pred_box[1]
                    width = pred_box[2] - pred_box[0]
                    height = pred_box[3] - pred_box[1]
                    rect = Rectangle(xy, width, height, fill=False, color=c, linewidth=2)
                    
                    axs[i, j].add_patch(rect)

            idx += 1 

show_preds(score_thresh = 0.3)
plt.show()