"""Semantic Segmentation performed by a PSPNet."""
import os

import mxnet as mx
import numpy as np
from mxnet import image
from tqdm import tqdm

from src.models.pspnet import PSPNet
from src.predictors.predict import Predict


class SemanticSegmentation(Predict):
    """Perform the semantic segmentation on a dataset by using a PSPNet."""

    def __init__(self, model_path: str = 'models/PSPNet_resnet50_1_epoch.params',
                 images_info_path: str = 'data/annotations/mini_test.json', no_cuda: bool = True):
        Predict.__init__(self, images_info_path, no_cuda)
        self.model_path = model_path
        self.model = self._load_model()

    @property
    def classes_name(self) -> list:
        return ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower',
                'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
                'railroad',
                'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
                'wall-stone',
                'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
                'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
                'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
                'food-other-merged',
                'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

    @property
    def classes(self) -> list:
        """Category coco index"""
        return [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
                149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    def _load_model(self) -> PSPNet:
        """Load PSPNet and the trained parameters.

        Returns
        -------

        The model with the associate parameters.

        """
        model = PSPNet(nclass=53, backbone='resnet50', pretrained_base=False, ctx=self.ctx)
        model.load_parameters(self.model_path)
        return model

    def predict(self, img_path: str = 'data/test2017'):
        """Predict the semantic segmentation on a dataset.

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
        semantic_segmentation = []

        tbar = tqdm(self._imgs_idx)
        for idx in tbar:
            img_metadata = coco.loadImgs(idx)[0]
            path = img_metadata['file_name']
            img = image.imread(os.path.join(img_path, path))
            img = self.transform(img)
            img = img.expand_dims(0).as_in_context(self.ctx[0])
            # TODO: really need to change this....
            output = model.demo(img)
            predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
            predicted_categories = list(np.unique(predict))

            for category in predicted_categories:
                # TODO: I think the category 0 is not 'banner' as expected... Need to look at the training.
                if category == 0.0: continue
                binary_mask = (np.isin(predict, category) * 1)
                binary_mask = np.asfortranarray(binary_mask).astype('uint8')
                segmentation_rle = coco_mask.encode(binary_mask)
                result = {"image_id": int(idx),
                          "category_id": self.classes[int(category)],
                          "segmentation": segmentation_rle,
                          }
                semantic_segmentation.append(result)
        self.predictions = semantic_segmentation


if __name__ == "__main__":
    predictor = SemanticSegmentation()
    predictor.predict()
    predictor.save_predictions(destination_dir='data/predictions', filename='predictions_test.json')
