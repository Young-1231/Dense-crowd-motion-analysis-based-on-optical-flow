from utils.flow_viz import *
from utils.write_flow import *
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import os

# 光流计算方法 'f':farneback, 't':tvl1
method = "t"

# root
root_path = "D:/datasets/transfer1-1-20231231170000-20231231203000-100992192"
suffix = "png"
# output
output_path = "D:/datasets/output"

if not osp.exists(output_path):
    os.makedirs(output_path)


def compute_flow(img1, img2, method="f"):
    if method == "f":
        # 参数设置
        pyr_scale = 0.5
        levels = 3
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.1
        flags = 0

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            img1,
            img2,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
        )

    elif method == "t":
        # 创建 Dual TV-L1 光流对象
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        # 光流计算
        flow = tvl1.calc(img1, img2, None)

    elif method == "l":

        cuMat1 = cv2.cuda_GpuMat()
        cuMat2 = cv2.cuda_GpuMat()
        cuMat1.upload(img1)
        cuMat2.upload(img2)

        TVL1 = cv2.cuda_OpticalFlowDual_TVL1.create()
        cuFlow = TVL1.calc(cuMat1, cuMat2, None)

        flow = cuFlow.download()

    else:
        raise ValueError("method should be f or t")

    return flow


images1 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[1:]
images2 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[:-1]
images = sorted(images1 + images2)


image_list = []
for i in tqdm(range(len(images) // 2)):
    img1 = cv2.imread(images[2 * i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(images[2 * i + 1], cv2.IMREAD_GRAYSCALE)

    # 计算光流
    flow = compute_flow(img1, img2, method)

    # 保存光流
    write_flow(osp.join(output_path, f"flow_{i:06d}.flo"), flow)

    # 提取光流
    image = flow_to_image(flow)
    vi_flow = cv2.cvtColor(
        image, cv2.COLOR_RGB2BGR
    )  # OpenCV使用BGR格式，所以要转换一下
    # 显示结果
    cv2.imwrite(osp.join(output_path, f"flow_{i:06d}.png"), vi_flow)
