'''
Runs the wildcat/baseline classification models' training for pneumonia scans
'''
#import files from Nathan Densenet start
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from pydicom.contrib.pydicom_PIL import show_PL
import pathlib
from PIL import Image
from sklearn.metrics import roc_auc_score

#Pytorch packages
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
#import from Nathan Densenet ends

import argparse
from torchsummary import summary
import torch
import torch.nn as nn

from wildcat.engine import MulticlassEngine
from wildcat.chestxray import ChestXRay
from wildcat.models import resnet101_wildcat
from wildcat.models import densenet121_wildcat
from wildcat.models import vgg_wildcat
from wildcat.util import AveragePrecisionMeter, Warp
from sklearn.metrics import roc_auc_score
import glob

#reproducability
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--model-dir', default='./forPraveer/rsna-pneumonia-detection-challenge/Pneumothora_models', type=str, metavar='MODELPATH',
                    help='path to model directory (default: none)')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=1, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--maps', default=1, type=int,
                    metavar='N', help='number of maps per class (default: 1)')
parser.add_argument('--adam', default=0, type=int,
                    metavar='A', help='Use Adam loss (1) vs SGD (0)')
parser.add_argument('--round', default=0, type=int,
                    metavar='C', help='Use curated dataset (Not necessary)')
parser.add_argument('--wild', default=1, type=int,
                    metavar='w', help='Use Wildcat (1) versus baseline model (0)')
parser.add_argument('--dense', default=1, type=int,
                    metavar='w', help='Use ResNet (0), Densenet (1), or VGG (2) models')
parser.add_argument('--run', default=1, type=int,
                    metavar='r', help='Run number (Not necessary)')
parser.add_argument('--percents', default=100., type=float,
                    metavar='r', help='Percent of training/validation data to use (assumes csv file with % already split')

'''
Define standard binary classification models to test
'''
# # define a standard VGG19 classifier model                    
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        num_ftrs = self.vgg19.classifier[0].in_features
        self.vgg19.classifier = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.vgg19(x)
        return x

# define a standard Densenet121 classifier model
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.densenet121(x)
        return x

# define a standard ResNet101 classifier model
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.resnet101(x)
        return x

# runs the WILDCAT model
def main_chestxray():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # sets the model name for optimizer (empty for SGD)
    adam_str = 'adam_' if args.adam else ''
    num_classes = 2
    # load the model based on the argument (wildcat vs regular)
    #       0 = ResNet
    #       1 = Densenet
    #       2 = VGG
    if args.dense == 1:
        model_str = 'densenet'
        if args.wild == 1:
            model = densenet121_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        else:
            model = DenseNet121()
    elif args.dense == 0:
        model_str = 'resnet'
        if args.wild == 1:
            model = resnet101_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        else:
            model = ResNet101()
    else:
        model_str = 'vgg'
        if args.wild == 1:
            model = vgg_wildcat(num_classes, pretrained=True, kmax=args.k, alpha=args.alpha, num_maps=args.maps)
        else:
            model = VGG19()

    # give user info about the model (wildcat vs regular baseline)
    if args.wild == 1:
        print('classifier', model.classifier)
        print('spatial pooling', model.spatial_pooling)
    else:
        model_str += '_baseline'

    # load the train and validation dataset
    train_dataset = ChestXRay(args.data, 'Supervised_{}%.csv'.format(float(args.percents)))
    val_dataset = ChestXRay(args.data, 'Val_{}%.csv'.format(float(args.percents)))
    round_str = 'curated_double_split'

    # for pneumothorax
    # round_str = 'pneumothorax'

    num_classes = 2

    # set the model's save name
    mod_name = 'lr{}_lrp{}_{}epochs{}_k{}_maps{}_alpha{}_{}_{}_balanced{}_percent{}'.format(args.lr, args.lrp, adam_str, args.epochs, args.k, args.maps, args.alpha, model_str, round_str, args.run, args.percents)
    print(mod_name)
    
    model = model.cuda()
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    #       0 - SGD
    #       1 - Adam
    # load optimizer depending on model type and data
    if args.adam == 0:
        print("ITS SGD")
        if args.wild == 1:
            optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    else:
        print("ITS ADAM")
        if args.wild == 1:
            print("using config optim")
            optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp),
                                    lr=args.lr,
                                    betas=(0.9,0.999))
        else:
            print("using param")
            optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    betas=(0.9,0.999))

    # set the states and pass the model into the engine for learning
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume}
    state['difficult_examples'] = False
    state['save_model_path'] = args.model_dir + mod_name #'./expes/models/chestxray/'

    engine = MulticlassEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_chestxray()
