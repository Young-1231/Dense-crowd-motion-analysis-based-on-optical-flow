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

    flow_mask = np.abs(flow[:, :, 1]) > 0.1
    # flow_mask = np.linalg.norm(flow, axis=2) > 1e-3
    vertical_main_mask = np.ones(flow[:, :, 1].shape, dtype=bool)
    # vertical_main_mask = np.abs(flow[:, :, 1]) > np.abs(flow[:, :, 0])
    horizontal_main_mask = ~vertical_main_mask
    vertical_mask = flow[:, :, 1] > 0
    horizontal_mask = flow[:, :, 0] > 0
    flow_direction = np.zeros_like(flow)
    flow_direction[np.logical_and(flow_mask, vertical_mask), 1] = -1
    flow_direction[np.logical_and(flow_mask, ~vertical_mask), 1] = 1
    # flow_direction[np.logical_and(flow_mask,horizontal_mask), 0] = -1
    # flow_direction[np.logical_and(flow_mask,~horizontal_mask), 0] = 1
    # up [0,1] down [0,-1] left [1,0] right [-1,,0]

    # 计算flow区域与gt区域每一类的交并比，3为up，4为down，5为过滤掉的区域
    intersaction3 = np.logical_and(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
    union3 = np.logical_or(flow_direction[:, :, 1] == 1, gt[:, :, 1] == 1)
    intersaction4 = np.logical_and(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
    union4 = np.logical_or(flow_direction[:, :, 1] == -1, gt[:, :, 1] == -1)
    intersaction5 = np.logical_and(flow_direction[:, :, 1] == 0, gt[:, :, 1] == 0)
    # 计算mIOU
    mIOU_up = np.sum(intersaction3) / (np.sum(union3) + 1e-6)
    mIOU_down = np.sum(intersaction4) / (np.sum(union4) + 1e-6)
    MIoU = (mIOU_up + mIOU_down) / 2
    # 计算准确率
    pixel_true_positive = np.sum(intersaction3) + np.sum(intersaction4)+ np.sum(intersaction5)
    total_pixel = flow.shape[0] * flow.shape[1]
    acc = pixel_true_positive / total_pixel
    return  mIOU_up, mIOU_down, MIoU, acc

def main(flow_dir, gt_dir):
    # 分为上下两个区域，MIoU和acc为最终凝练的指标
    mIOU_list_up = []
    mIOU_list_down = []
    MIoU_list = []

    acc_list = []
    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))

    for i in tqdm(range(len(flows))):
        # flow = cv2.readOpticalFlow(flows[i])
        with open(flows[i], 'rb') as f:
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data, (w, h, 2))

        gt = np.load(gts[i], allow_pickle=True)
        mIOU_up, mIOU_down, MIoU, acc = compute_mIOU(flow, gt['data'])

        mIOU_list_up.append(mIOU_up)
        mIOU_list_down.append(mIOU_down)
        MIoU_list.append(MIoU)
        acc_list.append(acc)

    mIOU_mean_up = np.mean(mIOU_list_up)
    mIOU_mean_down = np.mean(mIOU_list_down)
    MIoU_mean = np.mean(MIoU_list)
    acc_mean = np.mean(acc_list)
    print('mIOU_up: {:.4f}, mIOU_down: {:.4f}, MIoU_mean: {:.4f}'.format( mIOU_mean_up, mIOU_mean_down, MIoU_mean))
    print('pixel acc: {:.4f}'.format(acc_mean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_dir', default='E:\\Wuhan_Metro\\transfer_flo',help="flow directory")
    parser.add_argument('--gt_dir', default='E:\\Wuhan_Metro\\transfer1-1',help="gt directory")
    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir)
