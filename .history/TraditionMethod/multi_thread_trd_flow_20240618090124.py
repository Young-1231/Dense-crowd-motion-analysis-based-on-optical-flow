import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.flow_viz import *
from utils.write_flow import *


def compute_flow(img1, img2, method="f"):
    if method == "f":
        pyr_scale = 0.5
        levels = 3
        winsize = 15
        iterations = 3
        poly_n = 5
        poly_sigma = 1.1
        flags = 0
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
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
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
        raise ValueError("method should be f, t, or l")
    return flow


def process_image_pair(index, img1_path, img2_path, output_path, method, visual=False):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    flow = compute_flow(img1, img2, method)
    flow_file = osp.join(output_path, f"flow_{index:06d}.flo")
    image_file = osp.join(output_path, f"flow_{index:06d}.png")
    write_flow(flow_file, flow)

    if visual:
        image = flow_to_image(flow)
        vi_flow = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_file, vi_flow)


def main():
    visual = True

    # 光流计算方法 'f':farneback, 't':tvl1, 'l':cuda_tvl1
    method = "t"

    # root
    root_path = "/root/autodl-tmp/a"
    suffix = "png"
    # output
    output_path = "/root/autodl-tmp/output"

    if not osp.exists(output_path):
        os.makedirs(output_path)

    # 获取所有图像对
    images1 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[1:]
    images2 = sorted(glob(osp.join(root_path, f"*.{suffix}")))[:-1]
    image_pairs = list(zip(images1, images2))

    print(osp.exists(root_path))
    print(len(image_pairs))

    # 并行处理图像对
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, (img1_path, img2_path) in enumerate(image_pairs):
            futures.append(
                executor.submit(
                    process_image_pair,
                    i,
                    img1_path,
                    img2_path,
                    output_path,
                    method,
                    visual,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # 获取结果，确保异常被抛出

    print("全部光流计算完成。")


if __name__ == "__main__":
    main()
