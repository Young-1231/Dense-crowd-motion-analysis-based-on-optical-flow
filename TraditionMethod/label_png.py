from flow_viz import *
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import os


def output_flow(_dir_path, _out_path, method='t', suffix='png'):
    """
    method: 'f':farneback, 't':tvl1
    """

    if not osp.exists(_out_path):
        os.makedirs(_out_path)

    def compute_flow(_img1, _img2, _method='f'):
        if _method == 'f':
            # 参数设置
            pyr_scale = 0.5
            levels = 3
            win_size = 15
            iterations = 3
            poly_n = 5
            poly_sigma = 1.1
            flags = 0

            # 计算光流
            _flow = cv2.calcOpticalFlowFarneback(_img1, _img2, None, pyr_scale, levels, win_size, iterations,
                                                 poly_n, poly_sigma, flags)
        elif _method == 't':
            # 创建 Dual TV-L1 光流对象
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            # 光流计算
            _flow = tvl1.calc(_img1, _img2, None)
        elif _method == 'l':
            cuMat1 = cv2.cuda_GpuMat()
            cuMat2 = cv2.cuda_GpuMat()
            cuMat1.upload(_img1)
            cuMat2.upload(_img2)

            TVL1 = cv2.cuda_OpticalFlowDual_TVL1.create()
            cuFlow = TVL1.calc(cuMat1, cuMat2, None)

            _flow = cuFlow.download()
        else:
            raise ValueError('method should be f or t')

        return _flow

    def write_flow(filename, uv, v=None):
        """ Write optical flow to file.

        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        TAG_CHAR = np.array([202021.25], np.float32)
        nBands = 2

        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert (u.shape == v.shape)
        height, width = u.shape
        f = open(filename, 'wb')

        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)

        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    images1 = sorted(glob(osp.join(_dir_path, f"*.{suffix}")))[1:]
    images2 = sorted(glob(osp.join(_dir_path, f"*.{suffix}")))[:-1]
    images = sorted(images1 + images2)

    for i in tqdm(range(len(images) // 2)):
        img1 = cv2.imread(images[2 * i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[2 * i + 1], cv2.IMREAD_GRAYSCALE)

        # 计算光流
        flow = compute_flow(img1, img2, method)

        # 保存光流
        write_flow(osp.join(_out_path, f'flow_{i+1:04d}.flo'), flow)

        # 提取光流
        image = flow_to_image(flow)
        vi_flow = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV使用BGR格式，所以要转换一下
        # 显示结果
        cv2.imwrite(osp.join(_out_path, f'flow_{i+1:04d}.png'), vi_flow)


root_path = 'E:/data/Wuhan_Metro/'
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.ts'):
            dir_path = os.path.join(root, file.split('.')[0])
            out_path = os.path.join(dir_path, 'flow')
            output_flow(dir_path, out_path, method='t', suffix='png')
