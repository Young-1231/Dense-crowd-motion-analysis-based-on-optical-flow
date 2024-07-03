# 计算光流图的方向数目与label中的方向数目MSE与MAE
# 可指定计算任意方向：上下左右
from collections import Counter
import numpy as np
import warnings
import os
from glob import glob
from tqdm import tqdm
import argparse
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = '5'

def compute_error(flow, gt, directions_to_include):
    # 统一大小
    if flow.shape != gt.shape:
        flow = cv2.resize(flow, (gt.shape[1], gt.shape[0]))

    # 初始化方向掩码
    flow_mask = np.linalg.norm(flow, axis=2) > 0.1
    flow_direction = np.zeros_like(flow)
    
    if 'up' in directions_to_include:
        vertical_mask = flow[:, :, 1] < 0
        flow_direction[np.logical_and(flow_mask, vertical_mask), 1] = 1
        
    if 'down' in directions_to_include:
        vertical_mask = flow[:, :, 1] > 0
        flow_direction[np.logical_and(flow_mask, vertical_mask), 1] = -1
        
    if 'left' in directions_to_include:
        horizontal_mask = flow[:, :, 0] < 0
        flow_direction[np.logical_and(flow_mask, horizontal_mask), 0] = 1
        
    if 'right' in directions_to_include:
        horizontal_mask = flow[:, :, 0] > 0
        flow_direction[np.logical_and(flow_mask, horizontal_mask), 0] = -1

    # # 调试可视化
    # H, W, _ = flow.shape
    # color_map = np.zeros((H, W, 3), dtype=np.uint8)
    # for i in range(H):
    #     for j in range(W):
    #         if 'up' in directions_to_include and np.array_equal(flow_direction[i, j], [0, 1]):
    #             color_map[i, j] = [0, 0, 255]  # up: blue
    #         if 'down' in directions_to_include and np.array_equal(flow_direction[i, j], [0, -1]):
    #             color_map[i, j] = [0, 255, 0]  # down: green
    #         if 'left' in directions_to_include and np.array_equal(flow_direction[i, j], [1, 0]):
    #             color_map[i, j] = [255, 0, 0]  # left: red
    #         if 'right' in directions_to_include and np.array_equal(flow_direction[i, j], [-1, 0]):
    #             color_map[i, j] = [255, 255, 0]  # right: yellow

    # plt.figure(figsize=(10, 10))
    # plt.imshow(color_map)
    # plt.title('Flow Direction Visualization')
    # plt.axis('off')
    # plt.show()

    # 计算flow中的方向数
    u = flow_direction[:, :, 0]
    v = flow_direction[:, :, 1]
    directions = list(zip(u.flatten(), v.flatten()))
    direction_counter = Counter(directions)
    total_vectors = sum(direction_counter.values())
    most_common_directions = direction_counter.most_common()
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
    directions = list(zip(u.flatten(), v.flatten()))

    # 指定的可能方向向量
    possible_directions = []
    if 'up' in directions_to_include:
        possible_directions.append((0, 1))
    if 'down' in directions_to_include:
        possible_directions.append((0, -1))
    if 'left' in directions_to_include:
        possible_directions.append((1, 0))
    if 'right' in directions_to_include:
        possible_directions.append((-1, 0))

    # 检查每个可能的方向是否出现在光流图中
    present_directions = {direction: direction in directions for direction in possible_directions}

    # 统计出现的方向数目
    num_present_directions = sum(present_directions.values())
    # 计算MSE与MAE
    MSE = (direction_count - num_present_directions) ** 2
    MAE = np.abs(direction_count - num_present_directions)
    return MSE, MAE

def main(flow_dir, gt_dir, directions_to_include):
    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))

    MSE_list = []
    MAE_list = []

    for i in tqdm(range(len(flows))):
        flow = cv2.readOpticalFlow(flows[i])

        gt = np.load(gts[i], allow_pickle=True)
        MSE, MAE = compute_error(flow, gt['data'], directions_to_include)
        MSE_list.append(MSE)
        MAE_list.append(MAE)

    print("MSE: ", np.mean(MSE_list))
    print("MAE: ", np.mean(MAE_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', help="flow directory")
    parser.add_argument('--gt_dir', help="gt directory")
    parser.add_argument('--directions', nargs='+', default=['up', 'down', 'left', 'right'], help="directions to include: up, down, left, right")
    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir, args.directions)
