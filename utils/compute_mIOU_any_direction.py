# 计算flow区域与gt区域的交并比
# 可指定计算任意方向：上下左右
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import argparse

def compute_mIOU(flow, gt, directions_to_include):
    # 统一大小
    if flow.shape != gt.shape:
        flow = cv2.resize(flow, (gt.shape[1], gt.shape[0]))

    flow_mask = np.linalg.norm(flow, axis=2) > 0.1
    flow_direction = np.zeros_like(flow)

    if 'up' in directions_to_include:
        vertical_mask = flow[:, :, 1] > 0
        flow_direction[np.logical_and(flow_mask, vertical_mask), 1] = 1

    if 'down' in directions_to_include:
        vertical_mask = flow[:, :, 1] < 0
        flow_direction[np.logical_and(flow_mask, vertical_mask), 1] = -1

    if 'left' in directions_to_include:
        horizontal_mask = flow[:, :, 0] < 0
        flow_direction[np.logical_and(flow_mask, horizontal_mask), 0] = -1

    if 'right' in directions_to_include:
        horizontal_mask = flow[:, :, 0] > 0
        flow_direction[np.logical_and(flow_mask, horizontal_mask), 0] = 1

    # 计算每个方向的交并比
    mIOUs = {}
    for direction in directions_to_include:
        if direction == 'up':
            intersection = np.logical_and(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
            union = np.logical_or(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
        elif direction == 'down':
            intersection = np.logical_and(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
            union = np.logical_or(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
        elif direction == 'left':
            intersection = np.logical_and(flow_direction[:, :, 0] == -1, gt[:, :, 0] == -1)
            union = np.logical_or(flow_direction[:, :, 0] == -1, gt[:, :, 0] == -1)
        elif direction == 'right':
            intersection = np.logical_and(flow_direction[:, :, 0] == 1, gt[:, :, 0] == 1)
            union = np.logical_or(flow_direction[:, :, 0] == 1, gt[:, :, 0] == 1)

        mIOU = np.sum(intersection) / (np.sum(union) + 1e-6)
        mIOUs[direction] = mIOU

    # 计算总体的mIOU
    MIoU = np.mean(list(mIOUs.values()))

    # 计算准确率
    pixel_true_positive = sum(np.sum(np.logical_and(flow_direction[:, :, i] == gt[:, :, i], flow_direction[:, :, i] != 0)) for i in range(2))
    total_pixel = flow.shape[0] * flow.shape[1]
    acc = pixel_true_positive / total_pixel

    return mIOUs, MIoU, acc

def main(flow_dir, gt_dir, directions_to_include):
    mIOU_lists = {direction: [] for direction in directions_to_include}
    MIoU_list = []
    acc_list = []

    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))

    for i in tqdm(range(len(flows))):
        flow = cv2.readOpticalFlow(flows[i])

        gt = np.load(gts[i], allow_pickle=True)
        mIOUs, MIoU, acc = compute_mIOU(flow, gt['data'], directions_to_include)

        for direction in directions_to_include:
            mIOU_lists[direction].append(mIOUs[direction])
        MIoU_list.append(MIoU)
        acc_list.append(acc)

    mIOU_means = {direction: np.mean(mIOU_lists[direction]) for direction in directions_to_include}
    MIoU_mean = np.mean(MIoU_list)
    acc_mean = np.mean(acc_list)

    for direction, mIOU_mean in mIOU_means.items():
        print(f'mIOU_{direction}: {mIOU_mean:.4f}')
    print(f'MIoU_mean: {MIoU_mean:.4f}')
    print(f'pixel acc: {acc_mean:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', help="flow directory")
    parser.add_argument('--gt_dir', help="gt directory")
    parser.add_argument('--directions', nargs='+', default=['up', 'down', 'left', 'right'], help="directions to include: up, down, left, right")
    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir, args.directions)
