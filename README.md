# Panoptic Segmentation
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
The `environment.yaml` provided in this repo assume you have cuda 10.0 and will install the 
corresponding pytorch binaries.
```bash
conda env create -n panoptic_segmentation --file=environment.yaml
```

* By using docker  
