# 计算光流图的方向数目与label中的方向数目MSE与MAE
# 将flow方向区分为上下左右四个方向
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
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
    vertical_mask = flow[:, :, 1] > 0
    horizontal_mask = flow[:, :, 0] > 0
    flow_direction = np.zeros_like(flow)
    flow_direction[vertical_mask, 1] = 1
    flow_direction[~vertical_mask, 1] = -1
    flow_direction[horizontal_mask, 0] = 1
    flow_direction[~horizontal_mask, 0] = -1

    # 选择不同的聚类数目进行K-means聚类并计算轮廓系数
    davies_bouldin_scores = []
    K = range(1, 5)
    for k in K:
        if k == 1:
            centroid = np.mean(data, axis=0)
            sse = np.sum((data - centroid) ** 2)
            davies_bouldin_scores.append(sse)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
            labels = kmeans.labels_
            davies_bouldin_scores.append(davies_bouldin_score(data, labels))

    # 找到最佳的簇数即为光流方向数
    best_k = K[np.argmin(davies_bouldin_scores)]

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
    MSE = (best_k - num_present_directions) ** 2
    MAE = np.abs(best_k - num_present_directions)
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