'''
This file splits the total CSV files of data into train, validation, and test splits (90/10) first, then 90/10 of the first 90 split
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


def find_classes(dir):


    targets = ['No Pneumothorax','Pneumothorax']
    # assign a label to each class
    classes = dict()
    index=0
    for x in targets:
        classes[x]=index
        index+=1
    return classes


def write_csv_file(dir, images, labels, set):
    csv_file = os.path.join(dir, set + '.csv')
    #print(od_model)
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


def make_dataset(dir):

    image_names = []
    labels = []
    train_labels = os.path.join(dir,'data.csv')
    #read .csv file indicating
    df = pd.read_csv(train_labels)
    df = df.drop_duplicates(subset=['name'])

    X = df[['name']]
    y = df[['target']]

    '''
    Split the data into train and test sets;
    Define the size of the train and test sets; 0.1 gets a 90-10 train-test split;
    '''


    testsize = int(round(0.1*y.shape[0]))

    X_train_base, X_test, y_train_base, y_test = train_test_split(X, y, test_size=testsize, random_state=1,stratify=y)

    '''
    Split the data into train and validation sets;
    Define the size of the train and validation sets; 0.1 gets a 90-10 train-val split;
    '''


    val_test_size = int(round(0.1*y_train_base.shape[0]))

    X_train_original, X_val, y_train_original, y_val = train_test_split(X_train_base, y_train_base, test_size=val_test_size, random_state=1,stratify=y_train_base)


    print('Preparing train original')

    image_names = []
    labels = []
    for i, (ind,row) in enumerate(X_train_original.iterrows()):
        item = row['name']  # Tweak to use appropriate file directory
        image_names.append(item)

    for i, (ind, row) in enumerate(y_train_original.iterrows()):
        item = row['target']
        labels.append(item)

    write_csv_file(dir, image_names, labels, 'P_Supervised_100%')


    print('Preparing val')

    image_names = []
    labels = []
    for i, (ind,row) in enumerate(X_val.iterrows()):
        item = row['name'] # Tweak to use appropriate file directory
        image_names.append(item)

    for i, (ind, row) in enumerate(y_val.iterrows()):
        item = row['target']
        labels.append(item)

    write_csv_file(dir, image_names, labels, 'P_val')

    print('Preparing test')

    image_names = []
    labels = []
    for i, (ind,row) in enumerate(X_test.iterrows()):
        item = row['name']  # Tweak to use appropriate file directory
        image_names.append(item)

    for i, (ind, row) in enumerate(y_test.iterrows()):
        item = row['target']
        labels.append(item)

    write_csv_file(dir, image_names, labels, 'P_test')

    '''
    Split the training data into supervised and unsupervised training sets;
    Define the size of the sup & unsup; Currently choosing 1,10,25,50, 75 and 90%;
    '''
    splits = [0.005,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    #splits = [0.0001,0.001,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    for k in range(len(splits)):

        train_split_size = int(round((1-splits[k])*y_train_original.shape[0]))

        X_train_supervised, X_train_unsupervised, y_train_supervised, y_train_unsupervised = train_test_split(X_train_original, y_train_original, test_size=train_split_size, random_state=1,stratify=y_train_original)


        print('Preparing Supervised_'+str((splits[k]*100))+'%')

        image_names = []
        labels = []

        for i, (ind,row) in enumerate(X_train_supervised.iterrows()):
            item = row['name']# Tweak to use appropriate file directory
            image_names.append(item)

        for i, (ind, row) in enumerate(y_train_supervised.iterrows()):
            item = row['target']
            labels.append(item)

        writeset = 'P_Supervised_'+str((splits[k]*100))+'%'

        write_csv_file(dir, image_names, labels,writeset )

        print('Preparing Unsupervised_'+str((splits[k]*100))+'%')

        image_names = []
        labels = []

        for i, (ind,row) in enumerate(X_train_unsupervised.iterrows()):
            item = row['name'] # Tweak to use appropriate file directory
            image_names.append(item)

        for i, (ind, row) in enumerate(y_train_unsupervised.iterrows()):
            item = row['target']
            labels.append(item)

        writeset = 'P_Unsupervised_'+str((splits[k]*100))+'%'

        write_csv_file(dir, image_names, labels, writeset)

    '''
    Split the Validation data into same percentage as the corresponding supervised training sets;
    Define the size of the sup & unsup; Currently choosing .01, .05, .1, .5, 1, 5, 10, 25, 50, 75 and 90%;
    '''
    splits = [.005, .01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    #splits = [0.0001,0.001,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    for k in range(len(splits)):

        print(y_val.shape[0])
        print(1-splits[k])

        val_split_size = int(round((1-splits[k])*y_val.shape[0]))

        X_val_supervised, X_val_unsupervised, y_val_supervised, y_val_unsupervised = train_test_split(X_val, y_val, test_size=val_split_size, random_state=1,stratify=y_val)


        print('Preparing Val_'+str((splits[k]*100))+'%')

        image_names = []
        labels = []

        for i, (ind,row) in enumerate(X_val_supervised.iterrows()):
            item = row['name'] # Tweak to use appropriate file directory
            image_names.append(item)

        for i, (ind, row) in enumerate(y_val_supervised.iterrows()):
            item = row['target']
            labels.append(item)

        writeset = 'P_Val_'+str((splits[k]*100))+'%'

        write_csv_file(dir, image_names, labels,writeset )

if __name__ == '__main__':
    rootName = '../Pneumothorax_data'
    make_dataset(rootName)
