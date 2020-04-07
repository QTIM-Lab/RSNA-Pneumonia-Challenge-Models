import os
import sys
# from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import skimage.color
from tqdm import tqdm

import pydicom
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from utils.utils import TransformCfg
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pytorch_retinanet.dataloader import collater2d
import torch


# testDataset class, handles loading images as dicoms
class TestDataset(Dataset):
    """
    RSNA Challenge Pneumonia detection dataset, test patients   
    """

    def __init__(self, img_size: int, csv: str, img_path: str):
        """
        Args:
            img_size: the desired image size to resize to        
            csv : file path of csv file to run detection
        """
        super(TestDataset, self).__init__()  # inherit it from torch Dataset
        self.img_size = img_size

        # change csv file depending on if we want the curated or uncurated datasets
        self.samples = pd.read_csv(csv)

        self.patient_ids = list(sorted(self.samples.patientId.unique()))
        self.img_path = img_path

    def get_image(self, patient_id):
        """Load a dicom image to an array
        path altered for our dicom path """
        try:
            dcm_data = pydicom.read_file(self.img_path + f"{patient_id}.dcm")
            img = dcm_data.pixel_array
            return img

        except:
            pass

    def num_classes(self):
        return 1

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img = self.get_image(patient_id)
        img_h, img_w = img.shape[:2]

        # test mode augments
        cfg = TransformCfg(
            crop_size=self.img_size,
            src_center_x=img_w / 2,
            src_center_y=img_h / 2,
            scale_x=self.img_size / img_w,
            scale_y=self.img_size / img_h,
            angle=0,
            shear=0,
            hflip=False,
            vflip=False,
        )
        crop = cfg.transform_image(img)
        annotations = np.zeros((0,5))
        sample = {"img": crop, "scale": 1.0, "name": patient_id, "annot": annotations, "category": 1}
        return sample

# runs inference on the csv file
def infer(csv: str, output_csv: str, image_path: str, verbose=False):
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # load model checkpoint
    checkpoint = (
        "./models/detection_model.pt"
    )
    # initiate model and test data 
    if verbose:
        print("Loading model from checkpoint {}".format(checkpoint))

    model = torch.load(checkpoint)
    dataset_test = TestDataset(img_size=320, csv=csv, img_path=image_path)

    if verbose:
        print("Making testing dataset..")

    # load in the test dataset
    dataloader_test = DataLoader(
        dataset_test,
        num_workers=2,
        batch_size=1,
        shuffle=False,
        collate_fn=collater2d,
    )
    if verbose:
        print("Dataset size: {}\n".format(len(dataloader_test)))
    # initialize dataframe for storing the results
    df = pd.DataFrame(columns={"image", "x", "y", "width", "height", "score"})

    # make predictions
    for iter_num, data in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        img = data['img'].to(device).float()
        nms_scores, global_classification, transformed_anchors = model(
            img, return_loss=False, return_boxes=True
        )

        # if we have predictions, we want to sort these by the highest score
        # we choose the prediction with the highest score
        if len(nms_scores) > 0:
            zips = list(zip(nms_scores, transformed_anchors))
            sorted(zips, key=lambda x: x[0], reverse=True)
            scores, boxes = list(zip(*zips))
            score = scores[0]
            box = boxes[0]
            # once sorted, add these results to the dataframe
            df = df.append({
                "image": data["name"][0], 
                "x": int(box[0].item()), 
                "y": int(box[1].item()),
                "width": int(box[2].item()),
                "height": int(box[3].item()),
                "score": score.item()
            }, ignore_index=True)
        
        # else we have no predictions, so we just add 0s to the dataframe
        else:
            df = df.append({
                "image": data["name"][0], 
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "score": 0
            }, ignore_index=True)   

    if verbose:
        print("\nwriting DF with {} rows to {}".format(len(df), output_csv))

    # write the csv to the output_csv file
    df.to_csv('./results/' + output_csv, index=False, columns=['image', 'x', 'y', 'width', 'height', 'score'])

'''
 - input_csv contains the image names to test on
 - output_csv is the name of the output csv that we write to
 - image_path is the relative path to the dicom images
'''
input_csv = '../../../test2detect2.csv'
output_csv = 'output.csv'
image_path = '../../../stage_2_train_images/'

# Make the predictions, writes to results/ folder
infer(input_csv, output_csv, image_path, verbose=True)
print("Done!")
        
