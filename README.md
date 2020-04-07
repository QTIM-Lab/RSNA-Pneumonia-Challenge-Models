# RSNA-Pneumonia-Challenge-Models


## src
This folder contains the code used for detection and classification

### Detection
- To run, navigate to kaggle-rsna/
    - run `. create_env.sh` to create the environment 
- Change paths in `run_experiments.sh` and `config.py` to point to output, input csvs, and dicom images
* Assumes input includes csvs with 'name, x, y, height, width, class, Target' info
* Also assumes dicom image input
- Run `. run_experiments.sh` to train and test model
    - Only runs train and check_metric methods in `train_runner.py`
    - Edit src/datasets as necessary (only uses detection_dataset, dataset_valid, and test_dateset for train and check_metric)


### Classification
- Wildcat
    - Contains wildcat code to run wildcat models
    - Change `run_wildcat_m01.sh` as necessary to run different models
    - change and run `python3 class_grad.py` as appropriate to obtain AUC and precision results on the models
        - outputs a csv file with all of the results


## models
This folder contains pretrained models, stored in github LFS

### Detection
- positive_only_equalized_model
    - model trained on only positive pneumonia cases, where the dicom inputs are histogram equalized
    - performs the best so far (mAP on positive-only test cases is 0.251)
- 3_class_equalize_model
    - model trained on all 3 classes (Normal, Not Normal/Not Pneumonia, Pneumonia), dicoms, histogram-equalized
    - performs poorly (mAP 0.082)

### Classification
- Wildcat Models
    - Contains pretrained wildcat models used in the MICCAI paper