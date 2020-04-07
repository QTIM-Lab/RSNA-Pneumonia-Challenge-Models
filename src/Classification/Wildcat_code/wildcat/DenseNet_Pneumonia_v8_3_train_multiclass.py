'''
Multiclass classification densenet model
Unchecked/Untested, check with Praveer/Nathan for accuracy
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pydicom
#from pydicom.contrib.pydicom_PIL import show_PL
import pathlib
import pydicom
import imageio
from PIL import Image
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

root_dir = '/mnt/2017P001692_CT-Brain-Hemorrhage/Nathan/rsna-pneumonia-detection-challenge/'
#root_dir = '/home/ngaw/rsna-pneumonia-detection-challenge/'

#model name
model_name = 'densenet_pneumonia_v8'
train_labels = root_dir + 'stage_2_train_labels.csv'
#read .csv file indicating
df = pd.read_csv(train_labels)
df = df.drop_duplicates(subset=['patientId'])

X = df[['patientId']]
y = df[['Target']]


#split the data into train, validation, and test sets
val_test_size = round(0.1*y.shape[0]) #defines the size of the test and validation sets; 0.1 gets a 8/1/1 test split; 


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size, random_state=1,stratify=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_test_size, random_state=1,stratify=y)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_test_size, random_state=1,stratify=y_train)

#directory that stores all of the images
dataDir = root_dir + 'stage_2_train_images_PNG_resize320_para/'

#Location to save model
modelname = model_name + '.pth.tar'
datadir = root_dir + 'models/model_final/'

# Path for model checkpoints
checkpointDir = root_dir + 'models/model_checkpoint/'
checkpointName = model_name + '_checkpoint.pth.tar'

# Path of file for listing losses for each iteration
evaluationFilePath = root_dir + 'models/evaluation_docs/' 

#gpu device number
deviceNum = 1

# Batch size, maximum number of epochs (consistent with paper), number of classifications
trBatchSize = 16 
trMaxEpoch = 3
trNumClasses = 2
classNames = ["No Pneumonia","Pneumonia"]

# Class to create data set (image names + labels)
class ChestXRay(Dataset):
    def __init__(self, image_list_file, label_file, datadir, transform=None): # image_list_file refers to the csv file with images names + labels
        
        # Attributes 
        image_names = []
        labels = []

        for i, (ind, row) in enumerate(image_list_file.iterrows()):
            image_name = row['patientId'] + '.png' # Tweak to use appropriate file directory            
            image_names.append(dataDir + image_name)

        for i, (ind, row) in enumerate(label_file.iterrows()):
            label = row['Target']
            labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = imageio.imread(image_name)
        #make a [3,320,320] array to match with RGB input requirement
        #image = image.pixel_array
        #image = np.concatenate((image,image,image))
        image = Image.fromarray(image)
        image = image.convert('RGB')
        label0 = self.labels[index]
        label = torch.from_numpy(np.array(label0))
        
        #label = torch.FloatTensor(1,2) #gets the label into one_hot_output format
        #label[0][label0] = 1
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.image_names)

# TRANSFORM DATA
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Create list of transformations
transformList = []
transformList.append(transforms.Resize((320, 320)))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence_train=transforms.Compose(transformList) # Compose all these transformations (later apply to data set)

# Same thing but for validation and testdata
transformList = []
transformList.append(transforms.Resize((320, 320)))
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence_valid=transforms.Compose(transformList)
#transformSequence_test=transforms.Compose(transformList)

# LOAD DATASET
datasetTrain = ChestXRay(X_train, y_train, dataDir, transformSequence_train)
print(len(datasetTrain))
datasetValid = ChestXRay(X_val, y_val, dataDir, transformSequence_valid)
print(len(datasetValid))
# datasetTest = ChestXRay(X_test, y_test, dataDir, transformSequence_test)
# print(len(datasetTest))

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderValid = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
# dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)

# SET DEVICE 
device = torch.device(deviceNum if torch.cuda.is_available() else "cpu")

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 2)
            #nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.densenet121(x)
        return x

# TRAIN MODEL

# Send model to GPU
model = DenseNet121().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999)) # Apply same optimization settings as paper
criterion = torch.nn.CrossEntropyLoss() # Categorical Cross Entropy Loss (the loss function)

#Open the evaluation file
evaluationFile = open(evaluationFilePath + model_name + '.txt' , "w")

# Validation for checkpoints
numCheckpoints = 1
prevloss = 1

#Train the model
for epochID in range(0, trMaxEpoch):

    print("Epoch " + str(epochID + 1), end =" ")
    evaluationFile.write("Training Losses for Epoch " + str(epochID + 1) + " \n")

    running_loss_train = 0.0
    for batchcount, (varInput, target) in enumerate(dataLoaderTrain):
        print(batchcount, end=" ") 
        model.train()
        inputs = varInput.to(device)
        labels = target.to(device)
        optimizer.zero_grad() 
        with torch.set_grad_enabled(True): # enable gradient while training
            outputs = model(inputs)
            #outputs = outputs[:,1]
            trainloss = criterion(outputs, labels)
            trainloss.backward()
            optimizer.step()

        # Find validation loss for each iteration
        model.eval()
        valbatch, vallabel = iter(dataLoaderValid).next()
        with torch.set_grad_enabled(False): # don't change the gradient while validating
            val_inputs = valbatch.to(device)
            val_labels = vallabel.to(device)
            val_outputs = model(val_inputs)
            validloss = criterion(val_outputs, val_labels)

            evaluationFile.write("Batch " + str(batchcount) + " train loss = " + str(trainloss.item()) + " valid loss = " + str(validloss.item()) + "\n")
        
        running_loss_train += trainloss.item()*inputs.size(0)

        # Save checkpoints every 100 iterations
        if (batchcount + 1) % 100 == 0:
            print(">>Checkpoint " + str(numCheckpoints) + ": Saving at batch #" + str(batchcount) + "<<", end = " ") 
            evaluationFile.write("Checkpoint Saved: old valid loss = " + str(prevloss) + " new valid loss = " + str(validloss) + " \n")
            torch.save(model.state_dict(), checkpointDir + checkpointName + str(numCheckpoints) + '.pth.tar')
            numCheckpoints+=1
            prevloss = validloss

    # Check validation loss
    model.eval() # Evaluation mode
    running_loss_val = 0.0
    for batchcount, (varInput, target) in enumerate(dataLoaderValid):
        print(batchcount, end=" ")
        inputs = varInput.to(device)
        labels = target.to(device)
        with torch.set_grad_enabled(False): # don't change the gradient for validation
            outputs = model(inputs)
            validloss = criterion(outputs, labels)
        running_loss_val += validloss.item()*inputs.size(0) 

    epoch_loss_train = running_loss_train / len(datasetTrain)
    epoch_loss_val = running_loss_val / len(datasetValid)
    print('Train Loss: {:.4f} Val Loss: {:.4f}'.format(epoch_loss_train, epoch_loss_val))

evaluationFile.close()

# Save the model
torch.save(model.state_dict(), datadir+modelname)


# Get AUC on the validation set
all_labels = []
all_outputs = []
model.eval()
for batchcount, (varInput, target) in enumerate(dataLoaderValid):
    print(batchcount, end=" ")
    inputs = varInput.to(device)
    labels = target
    with torch.set_grad_enabled(False): # don't change the gradient
        outputs = model(inputs)

    all_labels = all_labels + list(np.squeeze(np.array(labels)))
    all_outputs = all_outputs + list(np.squeeze(np.array(outputs.cpu())))


for i in range(0,len(all_labels[0])): # for each pathology, so 15 AUCs computed
    true_labels = [] # labels for each image 
    outputs = [] # model's output for each image
    for label in all_labels:
        #print(label[i]) # add label for ONE pathology
        true_labels.append(label[i])
    for output in all_outputs:
        #print(output[i]) # add model output for ONE pathology
        outputs.append(output[i])

    # print the AUC for the pathology
    try: 
        auc = roc_auc_score(true_labels, outputs)
    except ValueError:
        auc = float('nan')
    print(classNames[i] + " AUC = " + str(auc))


# for i in range(0,len(all_labels[0])): # for each pathology, so 15 AUCs computed
#     true_labels = [] # labels for each image 
#     outputs = [] # model's output for each image
#     for label in all_labels:
#         #print(label[i]) # add label for ONE pathology
#         true_labels.append(label[i])
#     for output in all_outputs:
#         #print(output[i]) # add model output for ONE pathology
#         outputs.append(output[i])

#     # print the AUC for the pathology
#     try: 
#         auc = roc_auc_score(true_labels, outputs)
#     except ValueError:
#         auc = float('nan')
#     print(classNames[i] + " AUC = " + str(auc))

# true_labels = [] # labels for each image 
# outputs = [] # model's output for each image
# for label in all_labels:
#     #print(label[i]) # add label for ONE pathology
#     true_labels.append(label)
# for output in all_outputs:
#     #print(output[i]) # add model output for ONE pathology
#     outputs.append(output)

# # print the AUC for the pathology
# try: 
#     auc = roc_auc_score(true_labels, outputs)
# except ValueError:
#     auc = float('nan')
# print(classNames[0] + " AUC = " + str(auc))
