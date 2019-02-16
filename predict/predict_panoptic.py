from predict.predict_instance import PredictInstance
from predict.predict_semantic import PredictSemantic
import random
import numpy as np


class PanopticSeg:
    """Merge the instance segmentation predictions with the instance segmentation
    to output the panoptic segmentation. This process is described in the paper
    https://arxiv.org/abs/1801.00868. This code is reproducing the same
    process as https://github.com/cocodataset/panopticapi/blob/master/combine_semantic_and_instance_predictions.py

    Notes
    -----
    COCO Panoptic Segmentation format:

    annotation{
        "image_id"          : int,
        "file_name"         : str,
        "segments_info"     : [segment_info],
    }

    segment_info{
        "id"                : int,
        "category_id"       : int,
        "area"              : int,
        "bbox"              : [x,y,width,height],
        "iscrowd"           : 0 or 1,
    }

    categories[{
        "id"                : int,
        "name"              : str,
        "supercategory"     : str,
        "isthing"           : 0 or 1,
        "color"             : [R,G,B],
    }]


    Parameters
    ----------
    config_file: str

    min_img_size: int

    thresh: float

    frac: float

    Returns
    -------


    """

    def __init__(self, config_file: str, min_img_size: int = 800, thresh: float = 0.7, frac: float = 0.2):
        self._instance_segmentation = PredictInstance(config_file, min_img_size, thresh)
        self._semantic_segmentation = PredictSemantic()
        self._thresh = thresh
        self._frac = frac
        self._invert_mapping = {v: k for k, v in self._semantic_segmentation.classes.items()}

    def predict(self, img, img_id=0):
        """Predict panoptic segmentation on one image.
        """
        # We first compute the predictions for both instance and semantic segmentation
        # In the case of the instance segmentation, we have to re-organise the categories since
        # some of them were merged in the coco dataset:
        # 1-92 -> thing categories (80 total) <- no change
        # 92-182 -> original stuff categories (36 total) <- no change
        # 183-200 -> merged stuff categories (17 total) <- Change!
        boxes, mask, instance_labels, scores = self._predict_instance(img)
        labelmap, semantic_labels = self._predict_semantic(img)

        # Create random independents labels
        _ids = random.sample(range(1, 16711422), len(instance_labels) + len(
            semantic_labels[semantic_labels != -1]))
        ids_instance = _ids[:len(instance_labels)]
        ids_semantic = _ids[len(instance_labels):]

        RGB, canvas = self._merge_masks(
            img, semantic_labels, labelmap, mask, ids_semantic, ids_instance)

        buff = self._create_segments_info(
            canvas, boxes, mask, ids_semantic, ids_instance, semantic_labels, instance_labels)

        segment = {
            "segments_info": buff,
            "file_name": "{}.png".format(img_id),
            "image_id": img_id,
        }
        return segment, RGB

    def _predict_instance(self, img):
        """Predict both instance segmentation for a given single image and re-organise the classes.

        Parameters
        ----------
        img:

        Returns
        -------

        """
        bboxes, masks, labels, scores = self._instance_segmentation.predict(img)
        bbox, mask, label, score = bboxes[0], masks[0], labels[0], scores[0]

        # Filter and remap
        bbox, mask, label, score = self._instance_seg_filter(img, bbox, mask, label, score)
        label = np.array([self._invert_mapping[self._instance_segmentation.CATEGORIES[category]] for category in label],
                         dtype=label.dtype)
        return bbox, mask, label, score

    def _predict_semantic(self, img):
        """Predict the semantic segmentation for a given single image.

        Parameters
        ----------
        img

        Returns
        -------

        """
        labelmaps, labels = self._semantic_segmentation.predict(img)
        return labelmaps[0], labels[0]

    def _instance_seg_filter(self, img, bbox, mask, label, score):
        """Implement a part of the heuristic combinations described in  https://arxiv.org/abs/1801.00868
        section 7.

        Parameters
        ----------
        img
        bbox
        mask
        label
        score

        Returns
        -------

        """
        # If bbox is empty, nothing special to return.
        if len(bbox) == 0:
            return bbox, mask, label, score

        # As described int he paper, we consider only the top-performing instances
        # so first we are sorting by score.
        bbox, mask, label, score = map(np.array, list(
            zip(*sorted(zip(bbox, mask, label, score), key=lambda x: x[3], reverse=True))))

        # Then we apply a threshold to extract most confident predictions
        # The threshold parameter can be adjusted.
        # TODO: implement the threshold so we can perform grid search to find the best one.
        bests_predictions = score >= self._thresh
        bbox, mask = bbox[bests_predictions], mask[bests_predictions]
        label, score = label[bests_predictions], score[bests_predictions]

        # Now we perform a non-maximum suppression-like procedure.
        already_masked = np.full(img.shape[:2], True)
        frac_remain = np.full(score.shape, True)

        for idx, m in enumerate(mask):
            proposed_mask = already_masked & m
            remaining_fraction = np.sum(proposed_mask) / np.sum(m)
            if remaining_fraction < self._frac:
                frac_remain[idx] = False
            else:
                already_masked = already_masked & ~m

        return bbox[frac_remain], mask[frac_remain], label[frac_remain], score[frac_remain]

    @staticmethod
    def id_to_color(x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return x % 256, x % 256 ** 2 // 256, x // 256 ** 2

    @staticmethod
    def mask_to_bbox(a):

        x = np.any(a, axis=1)
        y = np.any(a, axis=0)
        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]

        return np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(int).tolist()

    @staticmethod
    def bbox_coco_format(bbox: list) -> np.array:
        """Convert the mask given by our model into the coco bounding box format.
        Coco bounding box format is [top left x position, top left y position, width, height].


        Parameters
        ----------
        bbox: list

        Returns
        -------
        np.array

        """
        bbox = np.array([
            bbox[:, 0],
            bbox[:, 1],
            bbox[:, 2] - bbox[:, 0],
            bbox[:, 3] - bbox[:, 1],
        ]).T
        return np.round(bbox).astype(int)

    def _merge_masks(self, img, labels_sema, labelmap, mask, ids_semantic, ids_instance):
        """Merge masks into one image with simple overlay, and translate into colors.

        """
        canvas = np.zeros(img.shape[:2])

        # TODO: smarter merge, if instance overlap > 80% (?) of semantic, merge into
        # one. When multiple, merge with closest (how?)
        # Or just delete if overlap > 80%?

        for idx, lab in enumerate(labels_sema[labels_sema != -1]):
            canvas[labelmap == lab] = ids_semantic[idx]

        for idx, m in enumerate(mask):
            canvas[m] = ids_instance[idx]

        RGB = np.zeros(img.shape, dtype=np.uint8)
        for u in np.unique(canvas):
            r, g, b = self.id_to_color(u)
            RGB[canvas == u, 0] = r
            RGB[canvas == u, 1] = g
            RGB[canvas == u, 2] = b
        return RGB, canvas

    def _create_segments_info(self, canvas, bbox, mask, ids_semantic, ids_instance, labels_sema, label):
        """Create segments outputs.

        """
        buff = []
        for idx, lab in enumerate(ids_semantic):
            m = canvas == lab
            _sum = np.sum(m).astype(int)
            if _sum == 0:
                # painted over by instance seg
                continue
            d = {
                "area": int(_sum),
                "category_id": int(labels_sema[labels_sema != -1][idx] + 1),
                "iscrowd": 0,
                "id": lab,
                "bbox": self.mask_to_bbox(m)
            }
            buff.append(d)

        _deboxed = self.bbox_coco_format(bbox)

        for idx, lab in enumerate(ids_instance):
            m = canvas == lab
            _sum = np.sum(m).astype(int)
            if _sum == 0:
                continue
            d = {
                "area": int(np.sum(mask[idx]).astype(int)),
                "category_id": int(label[idx] + 1),
                "iscrowd": 0,  # this parameter is not used in panoptic seg
                "id": lab,
                "bbox": _deboxed[idx].tolist()
            }
            buff.append(d)

        return buff
