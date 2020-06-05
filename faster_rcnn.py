from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import math

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

# function to train the faster rcnn
def train_model(model, train_loader, device, optimizer, scheduler, n_epochs):
    for epoch in range(n_epochs):
        epoch_losses = []
        all_losses = []
        for i, data in enumerate(train_loader, 0):
            inputs, targets, details, file_name = data
            
            # convert the intputs and targets to tensors to send to GPU
            inputs = [img.to(device) for img in inputs]
            targets = [{'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in targets]
            
            # first train the model
            model.train()
            
            optimizer.zero_grad()
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            epoch_losses.append(losses_reduced.item())

            losses.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % 100 == 0:
                print("Epoch %s, Progress: [%s/%s]" % (epoch+1, i, len(train_loader)))

        print("Epoch %s, Training Loss: %.4f" % (epoch+1, np.average(epoch_losses)))
        all_losses.append(np.average(epoch_losses))

        # save the trained model
        torch.save(model.state_dict(), "faster_rcnn.pt")

    return all_losses

# function to get the predictions of the faster rcnn
def get_preds(model, data_loader, device):
    model.eval()
    file_names = []
    preds = {}
    targets = {}
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, target, details, file_name = data
            
            # convert the intputs and targets to tensors to send to GPU
            inputs = [img.to(device) for img in inputs]
            target = [{'boxes':d['boxes'].to(device), 'labels':d['labels'].to(device)} for d in target]

            preds_list = model(inputs)

            d = {} 
            for j in range(len(preds_list)):
                # convert the tensors to lists
                preds_list[j]['boxes'] = preds_list[j]['boxes'].cpu().numpy().tolist()
                preds_list[j]['labels'] = preds_list[j]['labels'].cpu().numpy().tolist()
                preds_list[j]['scores'] = preds_list[j]['scores'].cpu().numpy().tolist()
                d['boxes'] = target[j]['boxes'].cpu().numpy().tolist()
                d['labels'] = target[j]['labels'].cpu().numpy().tolist()

                preds[file_name[j]] = preds_list[j]
                targets[file_name[j]] = target[j]

            file_names.extend(file_name)

            if i % 10 == 0:
                print("Progress: [%s/%s]" % (i, len(data_loader)))

    return preds, targets, file_names 

# Setup the paths
train_path = './data/cars_train/'
test_path = './data/cars_test/'
devkit_path = './data/cars_devkit'

train_annos_path = devkit_path + '/cars_train_annos.csv'
test_annos_path = devkit_path + '/cars_test_annos.csv'
cars_meta_path = devkit_path + '/cars_meta.csv'

# Define the transforms
transform = transforms.Compose([
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# Load in the Dataset
cars_data_train = CarsDataset('train_annos_cleaned.csv', train_path, transform=transform)

# Load in the indices used for the training and validation sets
subset_indices_train = np.load("train_indices.npy")
subset_indices_valid = np.load("valid_indices.npy")

# Collate function used for the Dataloader to keep data in the correct format
def collate_fn(batch):
    image = [item[0] for item in batch]
    target = [item[1] for item in batch]
    detail = [item[2] for item in batch]
    file_names = [item[3] for item in batch]
    return [image, target, detail, file_names]

# Setup the dataloaders
train_loader = torch.utils.data.DataLoader(cars_data_train, batch_size=1, 
                sampler=SubsetRandomSampler(subset_indices_train), collate_fn = collate_fn)
val_loader = torch.utils.data.DataLoader(cars_data_train, batch_size=1,
                sampler=SubsetRandomSampler(subset_indices_valid), collate_fn = collate_fn)

# load the pretrained Faster RCNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

# replace the classifier with a new one, that has
# num_classes which in the dataset + the background (0)
num_classes = 197
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# train the model
training_losses = train_model(model, train_loader, device, optimizer, lr_scheduler, 10)

# get training predictions
train_preds, train_targets, train_file_names = get_preds(model, train_loader, device)

# save training preds and targets(overwrites any previous predictions!)
with open('train_preds.json','w') as f:
    json.dump(train_preds,f)
with open('train_targets.json','w') as f:
    json.dump(train_targets,f)
np.save('train_file_names.npy',train_file_names)

val_preds, val_targets, val_file_names = get_preds(model, val_loader, device)

# save validation preds and targets(overwrites any previous predictions!)
with open('val_preds.json','w') as f:
    json.dump(val_preds,f)
with open('val_targets.json','w') as f:
    json.dump(val_targets,f)
np.save('val_file_names.npy',val_file_names)