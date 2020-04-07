'''
This file gets the test AUC/Precision on the wildcat models (includes outputs for validation as well)
Writes the results to a csv file
**** CHANGE LINES 43, 46-57 FOR TESTING DIFFERENT MODELS AND SAVING DIFFERENT CSV FILES *****
'''

import torch
from torch.autograd import Variable, Function
from torchvision import models, utils
import torchvision
import cv2
import sys
import numpy as np
from wildcat.models import densenet121_wildcat as DN121 
from wildcat.models import resnet101_wildcat as RN101
from wildcat.models import vgg_wildcat as VGG 
import matplotlib.patches as patches
import argparse
import matplotlib.pyplot as plt 
from skimage.transform import resize
from PIL import Image
import pandas as pd 
import imageio
from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, precision_score
from scipy.special import softmax
import torch.nn as nn
import glob

root_dir = ''

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set this if using pneumonia (from Praveer's files), else False for pneumothorax
# Pneumonia = True
Pneumonia = False
# SET THIS STRING TO POINT TO THE MODELS TO RUN TEST ON
    # STRING: the models that we are trying to run testing on
    # CSV NAME: csv file to write results to
    # DATA_PATH: Path that holds the csv files for validation/test
    # DATADIR: Path that holds the images 
if Pneumonia:
    string = './percent_models/*0.0001*/'
    csv_name = root_dir + 'percents.csv'
    data_path = '../../../../SemiSupervised/CSV_files_stage_2_train_labels_with_2_classes/'
    datadir =  '../stage_2_train_images_PNG_resize320_para/'


else:
    string = '../../../../../Bryan/wildcat_models/*/'
    csv_name = root_dir + '../../../../../Bryan/wildcat_results/baseline_lr_results.csv'
    data_path = '../../../../../Bryan/'
    datadir = data_path + 'Pneumothorax_Data/PNGs/train/'


# create the simple classificaiton models
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.resnet101(x)
        return x

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        num_ftrs = self.vgg19.classifier[0].in_features
        self.vgg19.classifier = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.vgg19(x)
        return x

# Class to create data set (image names + labels)
class ChestXRay():
    def __init__(self, image_list_file, label_file, datadir, transform=None): # image_list_file refers to the csv file with images names + labels
        
        # Attributes 
        image_names = []
        labels = []

        for i, (ind, row) in enumerate(image_list_file.iterrows()):
            image_name = row['name'] # Tweak to use appropriate file directory            
            image_names.append(datadir + image_name)

        for i, (ind, row) in enumerate(label_file.iterrows()):
            try:
                label = row['target']
            except: 
                label = row['label']
            labels.append(label)

        print("Verify that these are the same length! {}, {}".format(len(image_names), len(labels)))
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    # loads the image and label
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = imageio.imread(image_name)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        label0 = self.labels[index]
        label = torch.from_numpy(np.array(label0))
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.image_names)

def get_AUC_score(model, dataLoaderTest):
    classNames = ['No Pneumonia', 'Pneumonia']
    # Get AUC on the validation set
    all_labels = []
    all_outputs = []
    
    result = ''
    model.to(device)
    model.eval()
    for batchcount, (varInput, target) in enumerate(dataLoaderTest):
        print(batchcount, end=" ")
        inputs = varInput.to(device)
        labels = target
        with torch.set_grad_enabled(False): # don't change the gradient
            outputs = model(inputs)

        all_labels = all_labels + list(np.squeeze(np.array(labels)))
        all_outputs = all_outputs + list(np.squeeze(np.array(outputs.cpu())))


    for i in range(0,2): # for each pathology, so 15 AUCs computed
        true_labels = [] # labels for each image 
        outputs = [] # model's output for each image
        for label0 in all_labels:
            # add label for ONE pathology
            label = np.zeros((1,2))
            label[0, int(label0)]=1
            true_labels.append(label[0][i])
        for output0 in all_outputs:
            # add model output for ONE pathology
            output = softmax(output0)
            outputs.append(output[i])

        # print the AUC for the pathology
        try: 
            auc = roc_auc_score(true_labels, outputs)
        except ValueError:
            auc = float('nan')
    return auc, true_labels, [round(x) for x in outputs]
    
# get all of the models
models = glob.glob(string)
print(models)

# df2 stores the overall dataframe that will be saved to csv_name
df2 = pd.DataFrame(columns={'name', 'AUC_VAL', 'AUC_TEST', 'epoch'})

# iterate through the models, find auc and prec
for mod in models:
    print("working on ", mod)
    name = mod[len('./models/'):]

    # load the correct test and validation datasets
    if Pneumonia:
        try:
        # for pneumonia
            percent = float(name.split('percent')[1][:-1])
        except:
            percent = 100.0
        valid_labels = 'Val_{}%.csv'.format(percent)
        test_labels = 'test.csv'
    else:
        # for pneumothorax
        valid_labels = 'dataCreationCSVs/Val_{}%.csv'.format(float(100))
        test_labels = 'dataCreationCSVs/test.csv'

    df0 = pd.read_csv(data_path + valid_labels)
    df1 = pd.read_csv(data_path + test_labels)

    # prints values for user
    print("LENGTH OF VALID {} AND TEST {}".format(len(df0), len(df1)))
    target = 'label'

    # load validation and test data
    X_val = df0[['name']]
    y_val = df0[[target]]
    X_test = df1[['name']]
    y_test = df1[[target]]

    # figure out maps, k, and alpha based on model folder
    num_maps = int(mod.split("maps")[1].split('_')[0])
    k = int(float(mod.split("k")[1].split('_')[0]))
    alpha = float(mod.split("alpha")[1].split('_')[0])
    print("Number of maps: {}, k: {}, alpha:{}".format(num_maps, k, alpha))

    # load the models depending on the information from the folder
    if 'baseline' not in name:
        print('using wildcat')
        if 'densenet' in name:
            model = DN121(2, num_maps=num_maps, kmax=k, alpha=alpha)
        elif 'vgg' in name:
            model = VGG(2, num_maps=num_maps, kmax=k, alpha=alpha)
        else:
            model = RN101(2, num_maps=num_maps, kmax=k, alpha=alpha)
        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
    else:
        # otherwise, using baseline models
        print('using baseline')
        if 'densenet' in name:
            model = DenseNet121()
        elif 'vgg' in name:
            model = VGG19()
        else:
            model = ResNet101()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    print('loading models')
    # loads in the model
    tar = torch.load(mod + 'model_best.pth.tar')
    epoch = tar['epoch']
    state_dict = tar['state_dict']
    model.load_state_dict(state_dict)

    result = mod + "\n"

    # creating transforms for the data
    transformList = []
    transformList.append(transforms.Resize((320, 320)))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence_valid=transforms.Compose(transformList)
    transformSequence_test=transforms.Compose(transformList)

    # load the test and validation datasets
    datasetValid = ChestXRay(X_val, y_val, datadir, transformSequence_valid)
    dataLoaderValid = DataLoader(dataset=datasetValid, batch_size=24, shuffle=False, num_workers=24, pin_memory=True)
    print("dataloader length", len(dataLoaderValid))
    datasetTest = ChestXRay(X_test, y_test, datadir, transformSequence_test)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=24, shuffle=False, num_workers=24, pin_memory=True)


    model.eval()
    # get the auc scores of the model on both test and validation datasets
    v_auc, v_act, v_pred = get_AUC_score(model, dataLoaderValid)
    t_auc, t_act, t_pred = get_AUC_score(model, dataLoaderTest)

    # gets the precision score based on model outputs
    prec_v = precision_score(v_act, v_pred)
    prec_t = precision_score(t_act, t_pred)

    na = name.split('/')[0]
    # add the results to df2 the final dataframe
    df1 = pd.DataFrame({'name': name, 'precision_val': prec_v, 'precision_test': prec_t, 'AUC_VAL': v_auc, 'AUC_TEST': t_auc, 'epoch': epoch, "maps": num_maps}, index=[0])
    df2 = df2.append(df1, ignore_index=True)

    # show the data on this model
    print(df1.head())


def getNum(x):
    try:
        return int(x.split('baseline')[1][0])
    except:
        return 0

df2['percent'] = df2['name'].apply(lambda x: float(x.split('percent')[1][:-1]))
df2['run'] = df2['name'].apply(lambda x: getNum(x))
# save the dataframe, sorted by the name of the model
df2 = df2.sort_values(['percent', 'run'], ascending=[1, 1]) 
df2.to_csv(csv_name, index=False)

