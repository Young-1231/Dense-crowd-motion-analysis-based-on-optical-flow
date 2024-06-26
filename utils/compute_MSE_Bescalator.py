# 计算光流图的方向数目与label中的方向数目MSE与MAE
# 将flow方向区分为上下左右四个方向
from collections import Counter
import numpy as np
import warnings
import os
from glob import glob
from tqdm import tqdm
import argparse
import cv2

warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = '5'
def compute_error(flow, gt):
    # 统一大小
    if flow.shape != gt.shape:
        flow = cv2.resize(flow, (gt.shape[1], gt.shape[0]))
    # 获得主方向
    flow_mask = np.linalg.norm(flow, axis=2) > 1e-3
    vertical_main_mask = np.abs(flow[:, :, 1]) > np.abs(flow[:, :, 0])
    horizontal_main_mask = ~vertical_main_mask
    vertical_mask = flow[:, :, 1] > 0
    horizontal_mask = flow[:, :, 0] > 0
    flow_direction = np.zeros_like(flow)
    flow_direction[np.logical_and(np.logical_and(flow_mask,vertical_mask),vertical_main_mask), 1] = 1
    flow_direction[np.logical_and(np.logical_and(flow_mask,~vertical_mask),vertical_main_mask), 1] = -1
    flow_direction[np.logical_and(np.logical_and(flow_mask,horizontal_mask),horizontal_main_mask), 0] = 1
    flow_direction[np.logical_and(np.logical_and(flow_mask,~horizontal_mask),horizontal_main_mask), 0] = -1
    # 计算flow中的方向数
    u = flow_direction[:, :, 0]
    v = flow_direction[:, :, 1]
    directions = list(zip(u.flatten(), v.flatten()))
    # 统计每个方向向量的出现次数
    direction_counter = Counter(directions)
    # 计算总向量数目
    total_vectors = sum(direction_counter.values())
    # 按照数量从大到小排序
    most_common_directions = direction_counter.most_common()
    # 找到数量之和超过75%的方向
    threshold = 0.75 * total_vectors
    cumulative_count = 0
    direction_count = 0
    for direction, count in most_common_directions:
        cumulative_count += count
        direction_count += 1
        if cumulative_count >= threshold:
            break

    # 获得gt中的光流方向数
    u = gt[:, :, 0]
    v = gt[:, :, 1]
    # 合并u和v形成方向向量
    directions = list(zip(u.flatten(), v.flatten()))

    # 可能的四个方向向量
    possible_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 检查每个可能的方向是否出现在光流图中
    present_directions = {direction: direction in directions for direction in possible_directions}

    # 统计出现的方向数目
    num_present_directions = sum(present_directions.values())
    # 计算MSE与MAE
    MSE = (direction_count - num_present_directions) ** 2
    MAE = np.abs(direction_count - num_present_directions)
    return MSE, MAE

def main(flow_dir, gt_dir):
    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))

    MSE_list = []
    MAE_list = []

    for i in tqdm(range(len(flows))):
        flow = cv2.readOpticalFlow(flows[i])
        gt = np.load(gts[i], allow_pickle=True)
        MSE, MAE = compute_error(flow, gt['data'])
        MSE_list.append(MSE)
        MAE_list.append(MAE)

    print("MSE: ", np.mean(MSE_list))
    print("MAE: ", np.mean(MAE_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', help="flow directory")
    parser.add_argument('--gt_dir', help="gt directory")
    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir)