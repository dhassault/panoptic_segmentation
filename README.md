# Panoptic Segmentation
Time-constrained attempt to implement the paper: https://arxiv.org/abs/1801.00868  
2 different models are used to perform semantic segmentation and instance segmentation. 
Then the predictions are combined (by using an heuristic described in the paper) to adapt the format
of panoptic segmentation. Both models are compatible with Pytorch 1.0.

This implementation relies heavily on:  
- https://github.com/kazuto1011/deeplab-pytorch
- https://github.com/m3at/coco-panoptic
- https://github.com/facebookresearch/maskrcnn-benchmark
- https://github.com/cocodataset/panopticapi


## What is Panoptic Segmentation?
Panoptic segmentation addresses both stuff and thing classes, unifying the typically distinct semantic 
and instance segmentation tasks. The aim is to generate coherent scene segmentations 
that are rich and complete, an important step toward real-world vision systems such as in 
autonomous driving or augmented reality.  
For more details: http://cocodataset.org/#panoptic-2018
## Install
* By using conda  
The `environment.yaml` provided in this repo assumes you have `cuda 10.0` and will install the 
corresponding pytorch binaries.
```bash
conda env create -n panoptic_segmentation --file=environment.yaml
``` 

## Known issues
- Cannot run on CPU (need to be retrain by using pytorch 1.0)

## To do
- building a docker image
- using deeplabv3+ instead of deeplabv2
- retrain the models by using the right classes (instead of remapping)
- improving accuracy by retraining the models directly on the coco panoptic segmentation dataset