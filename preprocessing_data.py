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
Add label descriptions for each image file
'''
def label_details(row, cars_meta):
    index = row['label']
    return cars_meta.loc[index, 'label_details']

'''
Remove gray-scale images (~20 images)
'''
def rgb_images(data_path, df):
    for i, row in df.iterrows():
        file = row['file']
        image = Image.open(data_path + file)
        num_channel = len(image.split())
        if (num_channel != 3):
            df.loc[i, 'rgb'] = False
        else:
            df.loc[i, 'rgb'] = True
    df = df[df.rgb == True]
    df = df.drop('rgb', axis=1)
    return df

def main():
    
    '''
    Set paths to data and directories
    '''
    train_path = './data/cars_train/'
    test_path = './data/cars_test/'
    devkit_path = './data/cars_devkit'
    train_annos_path = devkit_path + '/cars_train_annos.csv'
    test_annos_path = devkit_path + '/cars_test_annos.csv'
    cars_meta_path = devkit_path + '/cars_meta.csv'
    
    print("Creating dataframes...")
    
    '''
    Create data frame with appropriate headers
    '''
    train_annos = pd.read_csv(train_annos_path, header=None)
    test_annos = pd.read_csv(test_annos_path, header=None)
    cars_meta = pd.read_csv(cars_meta_path, header=None)
    train_annos.columns = ['file', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'label']
    test_annos.columns = ['file', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'label']
    cars_meta.columns = ['label_details']
    train_annos['label'] = train_annos['label']-1
    test_annos['label'] = test_annos['label']-1
    
    print("Applying labels and headers...")
    
    train_annos['label_details'] = train_annos.apply(lambda row: label_details(row, cars_meta), axis=1)
    test_annos['label_details'] = test_annos.apply(lambda row: label_details(row, cars_meta), axis=1)
    
    print("Filtering out grayscale images...")
    
    train_annos = rgb_images(train_path, train_annos)
    test_annos = rgb_images(test_path, test_annos)
    
    print("Saving cleaned annotations...")
    
    train_annos.to_csv('train_annos_cleaned.csv', index=False)
    test_annos.to_csv('test_annos_cleaned.csv', index=False)

    print("DONE!")
    
if __name__ == "__main__":
    main()