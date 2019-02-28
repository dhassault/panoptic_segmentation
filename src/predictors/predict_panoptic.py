import copy
import json
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as COCOmask
from tqdm import tqdm

from src.panopticapi.utils import IdGenerator, id2rgb


class PanopticSegmentation:
    def __init__(self):
        pass

    def combine_to_panoptic(self, img_ids, img_id2img, inst_by_image,
                            sem_by_image, segmentations_folder, overlap_thr,
                            stuff_area_limit, categories):
        panoptic_json = []
        id_generator = IdGenerator(categories)
        tbar = tqdm(img_ids)
        for img_id in tbar:
            img = img_id2img[img_id]

            pan_segm_id = np.zeros((img['height'],
                                    img['width']), dtype=np.uint32)
            used = None
            annotation = {}
            annotation['image_id'] = img_id
            annotation['file_name'] = img['file_name'].replace('.jpg', '.png')

            segments_info = []
            for ann in inst_by_image[img_id]:
                area = COCOmask.area(ann['segmentation'])
                if area == 0:
                    continue
                if used is None:
                    intersect = 0
                    used = copy.deepcopy(ann['segmentation'])
                else:
                    intersect = COCOmask.area(
                        COCOmask.merge([used, ann['segmentation']], intersect=True)
                    )
                if intersect / area > overlap_thr:
                    continue
                used = COCOmask.merge([used, ann['segmentation']], intersect=False)

                mask = COCOmask.decode(ann['segmentation']) == 1
                if intersect != 0:
                    mask = np.logical_and(pan_segm_id == 0, mask)
                segment_id = id_generator.get_id(ann['category_id'])
                panoptic_ann = {}
                panoptic_ann['id'] = segment_id
                panoptic_ann['category_id'] = ann['category_id']

                pan_segm_id[mask] = segment_id
                segments_info.append(panoptic_ann)

            for ann in sem_by_image[img_id]:
                mask = COCOmask.decode(ann['segmentation']) == 1
                mask_left = np.logical_and(pan_segm_id == 0, mask)
                if mask_left.sum() < stuff_area_limit:
                    continue
                segment_id = id_generator.get_id(ann['category_id'])
                panoptic_ann = {}
                panoptic_ann['id'] = segment_id
                panoptic_ann['category_id'] = ann['category_id']
                pan_segm_id[mask_left] = segment_id
                segments_info.append(panoptic_ann)

            annotation['segments_info'] = segments_info
            panoptic_json.append(annotation)
            Image.fromarray(id2rgb(pan_segm_id)).save(
                os.path.join(segmentations_folder, annotation['file_name'])
            )
        return panoptic_json

    def combine_predictions(self, semseg_json_file, instseg_json_file, images_json_file,
                            categories_json_file, segmentations_folder,
                            panoptic_json_file, confidence_thr, overlap_thr,
                            stuff_area_limit):
        start_time = time.time()

        with open(semseg_json_file, 'r') as f:
            sem_results = json.load(f)
        with open(instseg_json_file, 'r') as f:
            inst_results = json.load(f)
        with open(images_json_file, 'r') as f:
            images_d = json.load(f)
        img_id2img = {img['id']: img for img in images_d['images']}

        with open(categories_json_file, 'r') as f:
            categories_list = json.load(f)
        categories = {el['id']: el for el in categories_list}

        if segmentations_folder is None:
            segmentations_folder = panoptic_json_file.rsplit('.', 1)[0]
        if not os.path.isdir(segmentations_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
            os.mkdir(segmentations_folder)

        print("Combining:")
        print("Semantic segmentation:")
        print("\tJSON file: {}".format(semseg_json_file))
        print("and")
        print("Instance segmentations:")
        print("\tJSON file: {}".format(instseg_json_file))
        print("into")
        print("Panoptic segmentations:")
        print("\tSegmentation folder: {}".format(segmentations_folder))
        print("\tJSON file: {}".format(panoptic_json_file))
        print("List of images to combine is takes from {}".format(images_json_file))
        print('\n')

        inst_by_image = defaultdict(list)
        for inst in inst_results:
            if inst['score'] < confidence_thr:
                continue
            inst_by_image[inst['image_id']].append(inst)
        for img_id in inst_by_image.keys():
            inst_by_image[img_id] = sorted(inst_by_image[img_id], key=lambda el: -el['score'])

        sem_by_image = defaultdict(list)
        for sem in sem_results:
            if categories[sem['category_id']]['isthing'] == 1:
                continue
            sem_by_image[sem['image_id']].append(sem)

        imgs_ids_all = img_id2img.keys()
        panoptic_json = []
        combined_segmentations = self.combine_to_panoptic(imgs_ids_all, img_id2img, inst_by_image,
                                                          sem_by_image, segmentations_folder, overlap_thr,
                                                          stuff_area_limit, categories)
        panoptic_json.append(combined_segmentations)

        with open(images_json_file, 'r') as f:
            coco_d = json.load(f)
        coco_d['annotations'] = panoptic_json
        coco_d['categories'] = categories.values()

        coco_d_ = str(coco_d)
        with open(panoptic_json_file, 'w') as f:
            json.dump(coco_d_, f)

        t_delta = time.time() - start_time
        print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    semseg_json_file = 'data/annotations/semantic_segmentation_mini_test2017.json'
    instseg_json_file = 'data/annotations/instance_segmentation_mini_test2017.json'
    images_json_file = 'data/annotations/mini_test.json'
    categories_json_file = 'src/panopticapi/panoptic_coco_categories.json'
    segmentations_folder = 'data/results/'
    panoptic_json_file = 'data/annotations/panoptic_results_mini_test2017.json'
    confidence_thr = 0.5
    overlap_thr = 0.5
    stuff_area_limit = 64 * 64

    combinator = PanopticSegmentation()
    combinator.combine_predictions(semseg_json_file, instseg_json_file, images_json_file,
                                   categories_json_file, segmentations_folder,
                                   panoptic_json_file, confidence_thr, overlap_thr,
                                   stuff_area_limit)
