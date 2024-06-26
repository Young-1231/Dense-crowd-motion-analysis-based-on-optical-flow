# 计算flow区域与gt区域的交并比
# 将flow方向区分为上下左右四个方向，四个类别计算mIOU
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import argparse

def compute_mIOU(flow, gt):
    # 统一大小
    if flow.shape != gt.shape:
        flow = cv2.resize(flow, (gt.shape[1], gt.shape[0]))
    # 将flow向量转换为方向在上下、左右各有一个主方向
    flow_mask = np.linalg.norm(flow, axis=2) > 1e-3
    vertical_main_mask = np.abs(flow[:, :, 1]) > np.abs(flow[:, :, 0])
    horizontal_main_mask = ~vertical_main_mask
    vertical_mask = flow[:, :, 1] > 0
    horizontal_mask = flow[:, :, 0] > 0
    flow_direction = np.zeros_like(flow)
    flow_direction[np.logical_and(flow_mask,vertical_mask), 1] = 1
    flow_direction[np.logical_and(flow_mask,~vertical_mask), 1] = -1
    flow_direction[np.logical_and(flow_mask,horizontal_mask), 0] = 1
    flow_direction[np.logical_and(flow_mask,~horizontal_mask), 0] = -1
    # 计算flow区域与gt区域每一类的交并比
    intersaction1 = np.logical_and(flow_direction[:, :, 0] == 1, gt[:, :, 0] == 1)
    union1 = np.logical_or(flow_direction[:, :, 0] == 1, gt[:, :, 0] == 1)
    intersaction2 = np.logical_and(flow_direction[:, :, 0] == -1, gt[:, :, 0] == -1)
    union2 = np.logical_or(flow_direction[:, :, 0] == -1, gt[:, :, 0] == -1)
    intersaction3 = np.logical_and(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
    union3 = np.logical_or(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
    intersaction4 = np.logical_and(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
    union4 = np.logical_or(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
    # 计算mIOU
    mIOU1 = np.sum(intersaction1) / (np.sum(union1) + 1e-6)
    mIOU2 = np.sum(intersaction2) / (np.sum(union2) + 1e-6)
    mIOU3 = np.sum(intersaction3) / (np.sum(union3) + 1e-6)
    mIOU4 = np.sum(intersaction4) / (np.sum(union4) + 1e-6)
    # 计算准确率
    acc1 = np.sum(intersaction1) / np.sum(flow_direction[:, :, 0] == 1)
    acc2 = np.sum(intersaction2) / np.sum(flow_direction[:, :, 0] == -1)
    acc3 = np.sum(intersaction3) / np.sum(flow_direction[:, :, 1] == 1)
    acc4 = np.sum(intersaction4) / np.sum(flow_direction[:, :, 1] == -1)
    return mIOU1, mIOU2, mIOU3, mIOU4, acc1, acc2, acc3, acc4


def main(flow_dir, gt_dir):
    mIOU_list1 = []
    mIOU_list2 = []
    mIOU_list3 = []
    mIOU_list4 = []
    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []

    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))

    for i in tqdm(range(len(flows))):
        flow = cv2.readOpticalFlow(flows[i])
        gt = np.load(gts[i], allow_pickle=True)
        mIOU1, mIOU2, mIOU3, mIOU4, acc1, acc2, acc3, acc4 = compute_mIOU(flow, gt['data'])
        mIOU_list1.append(mIOU1)
        mIOU_list2.append(mIOU2)
        mIOU_list3.append(mIOU3)
        mIOU_list4.append(mIOU4)

        acc_list1.append(acc1)
        acc_list2.append(acc2)
        acc_list3.append(acc3)
        acc_list4.append(acc4)

    mIOU1 = np.mean(mIOU_list1)
    mIOU2 = np.mean(mIOU_list2)
    mIOU3 = np.mean(mIOU_list3)
    mIOU4 = np.mean(mIOU_list4)
    acc1 = np.mean(acc_list1)
    acc2 = np.mean(acc_list2)
    acc3 = np.mean(acc_list3)
    acc4 = np.mean(acc_list4)
    print('mIOU1: {:.4f}, mIOU2: {:.4f}, mIOU3: {:.4f}, mIOU4: {:.4f}'.format(mIOU1, mIOU2, mIOU3, mIOU4))
    print('acc1: {:.4f}, acc2: {:.4f}, acc3: {:.4f}, acc4: {:.4f}'.format(acc1, acc2, acc3, acc4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', help="flow directory")
    parser.add_argument('--gt_dir', help="gt directory")
    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir)
