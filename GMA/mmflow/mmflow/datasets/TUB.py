# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

@DATASETS.register_module()
class TUB(BaseDataset):
    """TUB dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""
        self.img1_dir = osp.join(self.data_root, 'images//IM01')
        self.img2_dir = osp.join(self.data_root, 'images//IM01')
        self.flow_dir = osp.join(self.data_root, 'gt_flow//IM01')

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
            flow_filenames) + 1

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
        for i in range(num_file - 1):
            data_info = dict(
                img_info=dict(
                    filename1=img1_filename[i],
                    filename2=img2_filename[i + 1]),
                ann_info=dict())
            self.data_infos.append(data_info)

    def load_ann_info(self, filename: Sequence[str],
                      filename_key: str) -> None:
        """Load information of annotation.

        Args:
            data_infos (list): data information.
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation key e.g. 'flow'.
        """
        num_files = len(filename)
        for i in range(num_files):
            self.data_infos[i]['ann_info'][filename_key] = filename[i]