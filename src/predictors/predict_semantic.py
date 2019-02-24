"""Semantic Segmentation performed by a PSPNet."""
import json
import os

from src.models.pspnet import PSPNet

import numpy as np
import mxnet as mx

from mxnet import image
from mxnet.gluon.data.vision import transforms
from pycocotools.coco import COCO
from pycocotools import mask
from tqdm import tqdm


class InstanceSegmentation:
    """Perform the semantic segmentation on a dataset by using a PSPNet."""

    def __init__(self, model_path: str = 'models/PSPNet_resnet50_1_epoch.params',
                 images_info_path: str = 'data/annotations/mini_test.json', no_cuda: bool = True):
        self._coco = COCO(images_info_path)
        self._coco_mask = mask
        self._imgs_idx = self._imgs_idx()
        self.model_path = model_path
        self.ctx = self._define_context(no_cuda)
        self.model = self._load_model()
        self.transform = self._make_transform()
        self.predictions = []

    @staticmethod
    def _define_context(no_cuda: bool = True) -> list:
        """Define the context of the computations (CPU or GPU).

        Parameters
        ----------
        no_cuda : bool
            If True, the computations will happen on the CPU.

        Returns
        -------
        A list of the available computation device.

        """
        if no_cuda:
            ctx = [mx.cpu(0)]
        else:
            ctx = [mx.gpu(0)]
        return ctx

    @property
    def classes(self) -> list:
        """Category names"""
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

    def _imgs_idx(self):
        return list(self._coco.imgs.keys())

    @staticmethod
    def _make_transform() -> mx.gluon.data.vision.transforms.Compose:
        """Transform the images.

        Returns
        -------
        Transformations applied on the images.

        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

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

    def save_predictions(self, destination_dir: str, filename: str, empty_memory: bool = True):
        """Convert the binary strings in utf-8 encoding and dump the predictions in a json file.

        Parameters
        ----------
        destination_dir : str
            The directory where will be saved the predictions.

        filename : str
            The name of the file that contains the predictions.

        empty_memory : bool
            Since the predictions list take a lot of place in the memory, when the json file is written,
            one can use this tag to remove it from the memory.

        Returns
        -------

        """
        for prediction in self.predictions:
            prediction['segmentation']['counts'] = prediction['segmentation']['counts'].decode("utf-8")

        with open(destination_dir + '/' + filename, 'w') as f:
            json.dump(self.predictions, f)
        print('The predictions were just saved in: {}'.format(destination_dir + '/' + filename))

        if empty_memory:
            self.predictions = []


if __name__ == "__main__":
    predictor = InstanceSegmentation()
    predictor.predict()
    predictor.save_predictions(destination_dir='data/predictions', filename='predictions_test.json')
