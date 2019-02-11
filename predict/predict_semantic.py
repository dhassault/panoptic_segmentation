import numpy as np
import pandas as pd
import cv2
import yaml
import torch
import torch.nn.functional as F
from torch import nn
from ..models.deeplabv2 import DeepLabV2
from ..models.msc import MSC
from addict import Dict


class PredictSemantic:
    """Use deeplabv2 pytorch implementation to predict stuff on images."""
    def __init__(self, labels_path='../data/labels_2.txt',
                 model_path='../models_parameters/deeplab_orig_cocostuff164k_iter100k.pth', use_cuda=True):
        self._config = Dict(yaml.load(open('../config/cocostuff164k.yaml')))
        self._device = self.get_device(use_cuda)
        self.classes = self.generate_classes(labels_path)
        self._model = self.setup_model(model_path, len(self.classes), train=True)
        self._map_to_merged = self._add_merged_stuff()

    @staticmethod
    def get_device(use_cuda=True):
        """

        Parameters
        ----------
        use_cuda

        Returns
        -------

        """
        cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        return device

    @staticmethod
    def deeplabv2_resnet101_msc(n_classes: int):
        """

        Parameters
        ----------
        n_classes

        Returns
        -------

        """
        return MSC(
            base=DeepLabV2(n_classes=n_classes,
                           n_blocks=[3, 4, 23, 3],
                           atrous_rates=[6, 12, 18, 24]),
            scales=[0.5, 0.75])

    @staticmethod
    def generate_classes(labels_path: str) -> dict:
        """

        Parameters
        ----------
        labels_path

        Returns
        -------

        """
        classes = {}
        with open(labels_path) as f:
            for label in f:
                label = label.rstrip().split("\t")
                classes[int(label[0])] = label[1].split(",")[0]
        return classes

    def setup_model(self, model_path: str, n_classes: int, train=True):
        """

        Parameters
        ----------
        model_path
        n_classes
        train

        Returns
        -------

        """
        model = self.deeplabv2_resnet101_msc(n_classes=n_classes)
        # if using last version of pytorch, can load only in GPU
        # TODO: Need to fine tune the model on the last version of pytorch
        # to be able to oad the model on the cpu
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if train:
            model.load_state_dict(state_dict, strict=False)  # to skip ASPP
            return nn.DataParallel(model)
        else:
            model.load_state_dict(state_dict)
            model = nn.DataParallel(model)
            model.eval()
            return model.to(self._device)

    def _add_merged_stuff(self):
        """

        Returns
        -------

        """
        # This model was not trained with the new stuff-merged, so map manually
        # using this string from http://cocodataset.org/#panoptic-eval
        # Also set to "VOID" (-1) deleted stuff.
        s = """\
        tree-merged: branch, tree, bush, leaves
        fence-merged: cage, fence, railing
        ceiling-merged: ceiling-tile, ceiling-other
        sky-other-merged: clouds, sky-other, fog
        cabinet-merged: cupboard, cabinet
        table-merged: desk-stuff, table
        floor-other-merged: floor-marble, floor-other, floor-tile
        pavement-merged: floor-stone, pavement
        mountain-merged: hill, mountain
        grass-merged: moss, grass, straw
        dirt-merged: mud, dirt
        paper-merged: napkin, paper
        food-other-merged: salad, vegetable, food-other
        building-other-merged: skyscraper, building-other
        rock-merged: stone, rock
        wall-other-merged: wall-other, wall-concrete, wall-panel
        rug-merged: mat, rug, carpet"""
        # Turn string into useful mapping
        map_into_merged_int = {vv: idx + 183 for idx, (k, v) in enumerate(
            x.split(": ") for x in s.split("\n")) for vv in v.split(", ")}
        # Add mapping for delete stuff
        map_into_merged_int.update({k: -1 for k in [
            "furniture-other", "metal", "plastic", "solid-other",
            "structural-other", "waterdrops", "textile-other", "cloth",
            "clothes", "plant-other", "wood", "ground-other"]})

        _inv = {v: k for k, v in self.classes.items()}
        _map_to_merged = {_inv[k]: v for k, v in map_into_merged_int.items()}

        extend_stuff_merged = {idx + 183: k for idx, (k, v) in enumerate(
            x.split(": ") for x in s.split("\n"))}
        self.classes.update(extend_stuff_merged)
        self.classes.update({-1: "VOID"})
        return _map_to_merged

    def _replace_labels_with_merged(self, labelmap):
        """

        Parameters
        ----------
        labelmap

        Returns
        -------

        """
        return pd.DataFrame(labelmap).replace(self._map_to_merged).values

    def _preprocess_one(self, img):
        image = img.copy().astype(float)
        scale = self._config.IMAGE.SIZE.TEST / max(image.shape[:2])
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        image -= np.array(
            [
                float(self._config.IMAGE.MEAN.B),
                float(self._config.IMAGE.MEAN.G),
                float(self._config.IMAGE.MEAN.R),
            ]
        )
        return image.transpose(2, 0, 1)

    def _preprocess_image(self, imgs):
        """

        Parameters
        ----------
        imgs

        Returns
        -------

        """
        buff = []
        for img in imgs:
            buff.append(self._preprocess_one(img))
        image = torch.from_numpy(np.array(buff)).float()
        return image.to(self._device)

    def predict(self, img):
        """

        Parameters
        ----------
        img

        Returns
        -------

        """
        if isinstance(img, np.ndarray) and img.ndim == 3:
            return self._predict_batch([img])
        return self._predict_batch(img)

    def _predict_batch(self, imgs):
        """

        Parameters
        ----------
        imgs

        Returns
        -------

        """
        image = self._preprocess_image(imgs)
        self._model.to(self._device)
        output = self._model(image)
        # 0.2s
        output = F.interpolate(
            output,
            size=imgs[0].shape[:2],
            mode="bilinear", align_corners=True
        )
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()

        labelmaps = np.argmax(output, axis=1)
        labelmaps = np.array([
            self._replace_labels_with_merged(x) for x in labelmaps])
        labels = np.array([np.unique(l) for l in labelmaps])
        return labelmaps, labels
