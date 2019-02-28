# Panoptic Segmentation
Time-constrained attempt to implement the paper: https://arxiv.org/abs/1801.00868  
2 different models are used to perform semantic segmentation and instance segmentation. 
Then the predictions are combined (by using an heuristic described in the paper) to adapt the format
of panoptic segmentation. Both models are implemented in MXNet.

Thanks to:
- https://github.com/cocodataset/panopticapi
- MXNet devs, the training code is mainly from them.

This implementation is using mainly:
- MXNet (for cuda10 and intel cpu in the requirements.txt, be careful)
- Pillow-SIMD
- pycocotools


## What is Panoptic Segmentation?
Panoptic segmentation addresses both stuff and thing classes, unifying the typically distinct semantic 
and instance segmentation tasks. The aim is to generate coherent scene segmentations 
that are rich and complete, an important step toward real-world vision systems such as in 
autonomous driving or augmented reality.  
For more details: http://cocodataset.org/#panoptic-2018

## Expected directories structure
```
├── config
├── data
├── dataloaders
├── environment.yml
├── maskrcnn-benchmark
├── models
├── models_parameters
├── notebooks
├── panoseg.egg-info
├── predict
├── README.md
├── requirements.txt
├── setup.py
├── train
└── utils
```

## Install
* By using conda  
The `environment.yaml` provided in this repo assumes you have `cuda 10.0` and will install the 
corresponding pytorch binaries.
```bash
conda env create -n panoptic_segmentation --file=environment.yaml
``` 

## Downloads
Please download the data directory there: https://drive.google.com/file/d/1g6ZPDMqgVwdS2Ntb1YvYO4FbKF52hgFK/view?usp=sharing  
And the models directory: https://drive.google.com/file/d/1BPO2d66gwiUCNQLuLT8IQnUFtiKsfXGA/view?usp=sharing  
Extract everything in the root directory of this project. To have the directories structures showed above.

## Models used
For the instance segmentation, this code uses the pre-trained mask-rcnn available on gluoncv as it is.
For the semantic segmentation, PSPNet (resnet50) is used and trained for 1 epoch only on coco stuff dataset (with the stuff panoptic classes).

## Known issues
- predict_panoptic.py is not yet working. The format is not exactly the right one. Please use the script
provided in the panopticapi repo. I will fix this asap.
- not sure what happened about the class 0 of PSPNet. I removed it for now.
- for the inference in the semantic segmentation, I'm using the method `demo`, 
it's very bad I know but I couldn't manage the huge arrays with mxnet... I'll improve it later.
- to convert the outputs of both semantic and instance segmentation, we have to use python2...


## To do
- building a docker image.
- improving accuracy by training on more epochs the semantic segmentation model.
- adding predictor for 1 single image (have to add a method in each predictor).
- adding visualisation.
