from utils.flow_viz import *
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import os

# 光流计算方法 'f':farneback, 't':tvl1
method = 't'

# root
root_path = 'D:/datasets/transfer1-1-20231231170000-20231231203000-100992192'
suffix = 'png'
# output
output_path = 'D:/datasets/output'

if not osp.exists(output_path):
    os.makedirs(output_path)
    
def compute_flow(img1, img2, method='f'):
    if method == 'f':
        # 参数设置
        pyr_scale = 0.5
        levels = 3
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.1
        flags = 0

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

    elif method == 't':
        # 创建 Dual TV-L1 光流对象
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        # 光流计算
        flow = tvl1.calc(img1, img2, None)

    elif method == 'l':

        cuMat1 = cv2.cuda_GpuMat()
        cuMat2 = cv2.cuda_GpuMat()
        cuMat1.upload(img1)
        cuMat2.upload(img2)

        TVL1 = cv2.cuda_OpticalFlowDual_TVL1.create()
        cuFlow = TVL1.calc(cuMat1, cuMat2, None)

        flow = cuFlow.download()		


    else:
        raise ValueError('method should be f or t')
    
    return flow

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

images1 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[1:]
images2 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[:-1]
images = sorted(images1 + images2)


image_list = []
for i in tqdm(range(len(images)//2)):
    img1 = cv2.imread(images[2 * i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(images[2 * i + 1], cv2.IMREAD_GRAYSCALE)

    # 计算光流
    flow = compute_flow(img1, img2, method)

    # 保存光流
    write_flow(osp.join(output_path, f'flow_{i:06d}.flo'), flow)

    # 提取光流
    image = flow_to_image(flow)
    vi_flow = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV使用BGR格式，所以要转换一下
    # 显示结果
    cv2.imwrite(osp.join(output_path, f'flow_{i:06d}.png'),vi_flow)

