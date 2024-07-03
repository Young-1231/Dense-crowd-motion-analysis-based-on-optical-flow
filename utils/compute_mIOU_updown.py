# 计算flow区域与gt区域的交并比
# 将flow方向区分为上下左右四个方向，四个类别计算mIOU
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def compute_mIOU(flow, gt, img):
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

    if img != None:
        # 调试可视化
        H, W, _ = flow.shape
        color_map = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                # 只有上下两个方向
                if np.array_equal(flow_direction[i, j], [0, 1]):
                    color_map[i, j] = [0, 0, 255]  # up: blue
                if np.array_equal(flow_direction[i, j], [0, -1]):
                    color_map[i, j] = [0, 255, 0]  # down: green
                # if np.array_equal(flow_direction[i, j], [1, 0]):
                #    color_map[i, j] = [255, 0, 0]  # left: red
                # if np.array_equal(flow_direction[i, j], [-1, 0]):
                #    color_map[i, j] =[255, 255, 0]  # right: yellow

        image = cv2.imread(img)
        # 将掩码图像叠加在原始图像上
        alpha = 0.5  # 透明度
        vis_image = cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)
    else:
        vis_image = None


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
    return  mIOU_up, mIOU_down, MIoU, acc, vis_image

def main(flow_dir, gt_dir, image_dir, output_folder):
    # 分为上下两个区域，MIoU和acc为最终凝练的指标
    mIOU_list_up = []
    mIOU_list_down = []
    MIoU_list = []

    acc_list = []
    flows = sorted(glob(os.path.join(flow_dir, '*.flo')))
    gts = sorted(glob(os.path.join(gt_dir, '*.pkl')))
    imgs = None
    if image_dir is not None:
        imgs = sorted(glob(os.path.join(image_dir, '*.png')))
        # 保存结果图像到指定文件夹
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)

    for i in tqdm(range(len(flows))):
        # flow = cv2.readOpticalFlow(flows[i])
        with open(flows[i], 'rb') as f:
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data, (w, h, 2))

        # 读取图像文件（如果有）
        img = None
        if imgs is not None:
            img = imgs[i]

        gt = np.load(gts[i], allow_pickle=True)
        mIOU_up, mIOU_down, MIoU, acc, vis_image = compute_mIOU(flow, gt['data'], img)

        if image_dir is not None:
            output_path = os.path.join(output_folder, f'frame_{i:04d}.png')
            cv2.imwrite(output_path, vis_image)

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
    parser.add_argument('--image_dir',
                        default='E:\\Wuhan_Metro\\transfer_image',
                        help="image directory for visualization")
    parser.add_argument('--output_folder',
                        default='E:/Wuhan_Metro/transfer_vis',
                        help="image directory for visualization")
    # image_dir 和 output_folder请同时指定

    args = parser.parse_args()

    main(args.flow_dir, args.gt_dir, args.image_dir, args.output_folder)
