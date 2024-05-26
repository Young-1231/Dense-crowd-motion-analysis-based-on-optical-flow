# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class TUB(BaseDataset):
    """FlyingChairs dataset.

    Args:
        split_file (str): File name of train-validation split file for
            FlyingChairs.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        # unpack FlyingChairs directly, will see `data` subdirctory.
        self.img1_dir = osp.join(self.data_root, 'images//IM02')
        self.img2_dir = osp.join(self.data_root, 'images//IM02')
        self.flow_dir = osp.join(self.data_root, 'gt_flow//IM02')

        # data in FlyingChairs dataset has specific suffix
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'
        self.flow_suffix = '.flo'

        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        flow_filenames = self.get_data_filename(self.flow_dir,
                                                self.flow_suffix)

        assert len(img1_filenames) == len(img2_filenames) == len(
            flow_filenames)+1

        self.load_img_info(img1_filenames, img2_filenames)
        self.load_ann_info(flow_filenames, 'filename_flow')

    def load_img_info(self, img1_filename: Sequence[str],
                      img2_filename: Sequence[str]) -> None:
        """Load information of image1 and image2.

        Args:
            img1_filename (list): ordered list of abstract file path of img1.
            img2_filename (list): ordered list of abstract file path of img2.
        """

        num_file = len(img1_filename)
        for i in range(num_file-1):
            if (not self.test_mode) or self.test_mode:
                data_info = dict(
                    img_info=dict(
                        filename1=img1_filename[i],
                        filename2=img2_filename[i+1]),
                    ann_info=dict())
                self.data_infos.append(data_info)

    def load_ann_info(self, filename: Sequence[str],
                      filename_key: str) -> None:
        """Load information of optical flow.

        This function splits the dataset into two subsets, training subset and
        testing subset.

        Args:
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation e.g. 'flow'.
        """
        num_files = len(filename)
        num_tests = 0
        for i in range(num_files):
            if (not self.test_mode and False) \
                    or (self.test_mode and True):
                self.data_infos[
                    i - num_tests]['ann_info'][filename_key] = filename[i]
            else:
                num_tests += 1


"""



@DATASETS.register_module()
class TUB(BaseDataset):
    # TUB CrowdFlow dataset.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_data_dir(self) -> None:
      #  Get the paths for images and optical flow.
        # only provide ground truth for training

        # In KITTI 2015, data in `image_2` is original data
        self.img1_dir = osp.join(self.data_root, 'images//IM01')
        self.img2_dir = osp.join(self.data_root, 'images//IM01')
        self.flow_dir = osp.join(self.data_root, 'gt_flow//IM01')

        self.img1_suffix = '_img1.png'
        self.img2_suffix = '_img2.png'
        self.flow_suffix = '_flow.flo'

    def load_data_info(self) -> None:
        # Load data information, including file path of image1, image2 and optical flow.
        self._get_data_dir()
        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        flow_filenames = self.get_data_filename(self.flow_dir,
                                                self.flow_suffix)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')
"""
