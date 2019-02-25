import os

import numpy as np
from gluoncv import model_zoo, data, utils
from tqdm import tqdm as tqdm

from src.models.mask_rcnn import MaskRCNN
from src.predictors.predict import Predict


class InstanceSegmentation(Predict):
    def __init__(self):
        Predict.__init__(self)
        self.model = self._load_model()

    @property
    def classes_name(self) -> list:
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    @property
    def classes(self) -> list:
        """Category coco index"""
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    def _load_model(self) -> MaskRCNN:
        """Load MAsk-RCNN pretrained on coco things.

        Returns
        -------

        The model with the associate parameters.

        """
        model = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True, ctx=self.ctx)
        return model

    def predict(self, img_path: str = 'data/test2017'):
        """Predict the instance segmentation on a dataset.

        Parameters
        ----------
        img_path : str
            The directory path where the images are stored.

        Returns
        -------

        """
        coco = self._coco
        coco_mask = self._coco_mask

        model = self.model

        # The results will be stored in a list, it needs memory...
        # TODO: flush the memory periodically
        instance_segmentation = []

        tbar = tqdm(self._imgs_idx)
        for idx in tbar:
            img_metadata = coco.loadImgs(idx)[0]
            path = img_metadata['file_name']
            x, orig_img = data.transforms.presets.rcnn.load_test(os.path.join(img_path, path), short=np.amin(
                np.array([img_metadata['height'], img_metadata['width']])))
            width, height = orig_img.shape[1], orig_img.shape[0]
            ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in model(x.as_in_context(self.ctx[0]))]
            masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores, thresh=0.5)

            for k in range(len(masks)):
                binary_mask = np.asfortranarray(masks[k]).astype('uint8')
                segmentation_rle = coco_mask.encode(binary_mask)

                result = {"image_id": int(idx),
                          "category_id": self.classes[int(ids[0])],
                          "segmentation": segmentation_rle,
                          "score": float(scores[k])
                          }
                instance_segmentation.append(result)
            self.predictions = instance_segmentation


if __name__ == "__main__":
    predictor = InstanceSegmentation()
    predictor.predict()
    predictor.save_predictions(destination_dir='data/predictions', filename='predictions_test.json')
