from __future__ import print_function, division
import os
import torch
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import pandas as pd
import json

# Define the dataset
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
    
def compute_counts(preds, targets, iou_thr, score_thr):
    # define counters for the number of true positives, false poistives, and false negatives
    TP = 0
    FP = 0
    FN = 0

    # loop through all of the files
    for pred_file, pred in preds.items():
        # get the ground truths for each file
        gts = targets[pred_file]

        # reset the counter for the number of predictions and true postives for the image
        num_preds = 0
        tp = 0

        # get the number of ground truths for the image
        num_gt = len(gts['labels'])
        # loop through each prediction for the image
        for j in range(len(pred['labels'])):

            # check if the score of the prediction exceeds the threshold
            if (pred['scores'][j] >= score_thr):

                # increment the number of predictions
                num_preds += 1

                # check to see if there are still ground truths to be matched
                if tp < num_gt:

                    # check to see if the label matches
                    if (pred['labels'][j] == gts['labels'][0]):

                        # compute the iou for the bounding boxes
                        iou = compute_iou(pred['boxes'][j], gts['boxes'][0])
                    
                        #check to see if the iou exceeds the threshold
                        if iou > iou_thr:
                            tp += 1

        # compute the number of false positives
        fp = num_preds - tp

        # compute the number of false negatives
        fn = num_gt - tp

        # update the overall counters
        TP += tp
        FP += fp
        FN += fn

    return TP, FP, FN

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    # calculate the area of each box
    area_1 = (box_1[2]-box_1[0])*(box_1[3]-box_1[1])
    area_2 = (box_2[2]-box_2[0])*(box_2[3]-box_2[1])

    # find the top left and bottom right corners of the intersection
    mins = np.amin([box_1,box_2],axis = 0)
    maxes = np.amax([box_1,box_2],axis = 0)

    i = np.concatenate((maxes[:2],mins[2:]), axis = None)
    
    # calculate the area of the intersection
    if (i[2] >= i[0]) and (i[3] >= i[1]):
        area_i = (i[2]-i[0])*(i[3]-i[1])
    else:
        area_i = 0

    # calculate the area of the union
    area_u = area_1 + area_2 - area_i

    # calculate iou
    iou = area_i/area_u

    assert (iou >= 0) and (iou <= 1.0)

    return iou

# function for getting the precision and recall
def get_pr(preds, targets, iou_thrs, score_thrs):
    precision = []
    recall = []
    for iou_thr in iou_thrs:
        tp = np.zeros(len(score_thrs))
        fp = np.zeros(len(score_thrs))
        fn = np.zeros(len(score_thrs))
        for i, score_thr in enumerate(score_thrs):
            tp[i], fp[i], fn[i] = compute_counts(preds, targets, iou_thr=iou_thr, score_thr=score_thr)

        # get the total number of predictions and ground truths
        n_preds = tp + fp
        n_gt = tp + fn

        precision.append(tp/n_preds)
        recall.append(tp/n_gt)

    return precision, recall

# Setup the paths
train_path = './data/cars_train/'
test_path = './data/cars_test/'
devkit_path = './data/cars_devkit'

train_annos_path = devkit_path + '/cars_train_annos.csv'
test_annos_path = devkit_path + '/cars_test_annos.csv'
cars_meta_path = devkit_path + '/cars_meta.csv'

# Load in the training and validation predictions and targets
train_file_names = np.load('train_file_names.npy')
with open('train_preds.json','r') as f:
    train_preds = json.load(f)
with open('train_targets.json','r') as f:
    train_targets = json.load(f)

val_file_names = np.load('val_file_names.npy')
with open('val_preds.json','r') as f:
    val_preds = json.load(f)
with open('val_targets.json','r') as f:
    val_targets = json.load(f)

# get the precision and recall for the training and validation sets
precision_train, recall_train = get_pr(train_preds, train_targets, iou_thrs = [0.75], score_thrs = np.arange(0,1,0.01))
precision_val, recall_val = get_pr(val_preds, val_targets, iou_thrs = [0.75], score_thrs = np.arange(0,1,0.01))

# plot training PR curve
for i in range(len(precision_train)):
    plt.figure(1)
    plt.plot(recall_train[i],precision_train[i])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Training Set')

# plot the validation PR curve
for i in range(len(precision_val)):
    plt.figure(2)
    plt.plot(recall_val[i],precision_val[i])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Validation Set')

plt.show()