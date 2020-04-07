'''
Load the chestXRay data (by different csv file)
'''
#import files from Nathan Densenet start
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from pydicom.contrib.pydicom_PIL import show_PL
import pathlib
from sklearn.metrics import roc_auc_score

#Pytorch packages
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
#import from Nathan Densent ends
import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import torch.utils.data as data
from PIL import Image

from wildcat import util

# reproducability
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def find_classes(dir):
    # for pneumonia
    targets = ['No Pneumonia','Pneumonia']

    # for pneumothorax 
    # targets = ['No Pneumothorax','Pneumothorax']

    # assign a label to each class
    classes = dict()
    index=0
    for x in targets:
        classes[x]=index
        index+=1
    return classes


def make_dataset(dir,set):

    image_names = []
    labels = []
    # for pnuemonia
    path = '../../../../SemiSupervised/CSV_files_stage_2_train_labels_with_2_classes/'
    train_labels = os.path.join(path,  set)

    # for pnuemothorax
    # train_labels = os.path.join(dir, set)
    # print("dataset path", train_labels)
    #read .csv file indicating
    df = pd.read_csv(train_labels)
    df = df.drop_duplicates(subset=['name'])

    target = 'label'
    X = df[['name']]
    y = df[[target]]



    '''
    File refers to a csv file for either train, valid, or test
    Data is already assumed to be pre-split, so we don't need to split further and instead load data as is
    '''

    # use for Pneumonia
    for i, (ind,row) in enumerate(X.iterrows()):
        if '.png' not in row['name']:
            item = 'stage_2_train_images_PNG_resize320_para/' + row['name'] + '.png' # Tweak to use appropriate file directory
        else:
            item = 'stage_2_train_images_PNG_resize320_para/' + row['name']
        image_names.append(item)

    for i, (ind, row) in enumerate(y.iterrows()):
        item = row[target]
        labels.append(item)

    # # # use for Pneumothorax
    # for i, (ind,row) in enumerate(X.iterrows()):
    #     item = '../Pneumothorax_Data/PNGs/train/' + row['name'] # Tweak to use appropriate file directory
    #     image_names.append(item)

    # for i, (ind, row) in enumerate(y.iterrows()):
    #     item = row['label']
    #     labels.append(item)


    return image_names,labels


def write_csv_file(dir, images, labels, set):
    csv_file = os.path.join(dir, set + '.csv')
    if not os.path.exists(csv_file):

        # write a csv file
        print('[dataset] write file %s' % csv_file)
        with open(csv_file, 'w') as csvfile:
            fieldnames = ['name', 'target']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for (x,y) in zip(images, labels):
                writer.writerow({'name': x, 'target': y})

        csvfile.close()


class ChestXRay(data.Dataset):

    def __init__(self, root, set, transform=None, target_transform=None): # image_list_file refers to the csv file with images names + labels

        self.root = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.path_images = os.path.join(self.root,'')#stage_2_train_images contains all train and val images
        print("info", self.root, self.set)
        #download(self.root)

        self.classes = find_classes(self.root)
        self.image_names, self.labels = make_dataset(self.root,self.set)

        print('[dataset] ChestXRay set=%s  number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.image_names)))

        # write_csv_file(self.root, self.image_names, self.labels, self.set)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.path_images, image_name)).convert('RGB')
        #image = pydicom.dcmread(os.path.join(self.path_images,image_name))
        #make a [3,320,320] array to match with RGB input requirement
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is  not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_names)

    def get_number_classes(self):
        return len(self.classes)
