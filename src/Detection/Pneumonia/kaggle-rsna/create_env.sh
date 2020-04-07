#!/bin/bash

pip install torch==0.4.1
pip install torchvision==0.3.0
pip install --upgrade pip
pip install -r requirements.txt
pip install pycocotools

cd src/pytorch_retinanet/lib
. build.sh
cd ../../..




