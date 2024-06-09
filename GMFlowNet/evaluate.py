import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import pad, grid_sample
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from core import datasets
import datasets
from utils import flow_viz
from utils import frame_utils

# from raft import RAFT, RAFT_Transformer
from core import create_model
from core.utils.metric import Interpolation_error, angular_error
from utils.utils import InputPadder, forward_interpolate

TRAIN_SIZE = [432, 960]

# def warp_image_with_flow(image, flow):
#     """
#     Use the flow field to warp the image
#     :param image: Input image, [batch_size, channels, height, width]
#     :param flow: Flow tensor, [batch_size, 2, height, width]
#     :return: Input image warped by the flow field
#     """
#     batch_size, channels, height, width = image.size()
#     grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
#     grid = torch.stack((grid_x, grid_y), 2).float().to(image.device)  # generate grid
#
#     grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
#     grid = grid + flow.permute(0, 2, 3, 1)  # apply flow
#
#     grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (width - 1) - 1.0
#     grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (height - 1) - 1.0
#
#     warped_image = F.grid_sample(image, grid, align_corners=True)
#     return warped_image

def warp_image_with_flow(image_array: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image using the given flow.

    Args:
        image_array (nparray): nparray of the image with the shape (H, W, 3).
        flow (nparray): The predicted optical flow with the shape (H, W, 2).

    Returns:
        image_ndarray (nparray): nparray of the image with the shape (H , W).
    """
    h, w = flow.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1) + flow
    flow_map = np.clip(flow_map, 0, np.array([w - 1, h - 1]))
    warped_image = np.zeros_like(image_array)
    for y in range(h):
        for x in range(w):
            new_x, new_y = flow_map[y, x].astype(int)
            warped_image[y, x] = image_array[new_y, new_x]
    return warped_image


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)
    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])
    return patch_weights

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, warm_start=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, is_validate=True)
        epe_list = []

        flow_prev, sequence_prev = None, None
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            sequence_prev = sequence

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


# @torch.no_grad()
# def validate_tub(model, sigma=0.05):
#     """
#     使用TUBCrowdFlow数据集进行验证
#     """
#     IMAGE_SIZE = [720, 1280]
#
#     hws = compute_grid_indices(IMAGE_SIZE)
#     weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
#
#     model.eval()
#     results = {}
#     for dstype in ['IM01', 'IM01_hDyn', 'IM02', 'IM02_hDyn', 'IM03', 'IM03_hDyn', 'IM04', 'IM04_hDyn', 'IM05',
#                    'IM05_hDyn']:
#         val_dataset = datasets.TubCrowdFlow(dstype=dstype)
#
#         epe_list = []
#         ae_list = []
#         ie_list = []
#
#         for val_id in tqdm(range(len(val_dataset)), desc=dstype):
#             image1, image2, flow_gt, _ = val_dataset[val_id]
#             image1 = image1[None].cuda()
#             image2 = image2[None].cuda()
#
#             flows = torch.zeros_like(flow_gt, device='cuda').unsqueeze(0)
#             flow_count = torch.zeros_like(flow_gt, device='cuda').unsqueeze(0)
#
#             for idx, (h, w) in enumerate(hws):
#                 image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
#                 image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
#
#                 flow_pre, _ = model(image1_tile, image2_tile, flow_init=None)
#
#                 # 假设这里的padding逻辑是正确的，根据实际情况调整
#                 padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
#
#                 # 对flow_pre中每个元素进行padding
#                 flow_pre_padded = [pad(flow_pre[i] * weights[idx], padding) for i in range(len(flow_pre))]
#                 weights_padded = pad(weights[idx], padding)
#
#                 # 将flow_pre_padded中每个元素加到flows上
#                 for i in range(len(flow_pre_padded)):
#                     flows += flow_pre_padded[i]
#                 flow_count += weights_padded
#
#             flow_pre = flows / flow_count
#             flow_pre = flow_pre[0].cpu()
#
#             # EPE
#             epe = torch.sum((flow_pre - flow_gt) ** 2, dim=0).sqrt()
#             epe_list.append(epe.view(-1).numpy())
#
#             # AE
#             dot_product = torch.sum(flow_pre * flow_gt, dim=0) + 1
#             norm_pred = torch.norm(flow_pre, dim=0) + 1
#             norm_gt = torch.norm(flow_gt, dim=0) + 1
#             cos_angle = dot_product / (norm_pred * norm_gt)
#             ae = torch.acos(cos_angle.clamp(-1, 1))
#             ae_list.append(ae.view(-1).numpy())
#
#             # IE
#             image1_cuda = image1[0]
#             image2_cuda = image2[0]
#             flow_pre_cuda = flow_pre.unsqueeze(0).cuda()
#             warped_image2 = warp_image_with_flow(image2_cuda.unsqueeze(0), flow_pre_cuda)
#             ie = torch.abs(image1_cuda - warped_image2[0]).mean().cpu()
#             ie_numpy = ie.view(-1).numpy()
#             ie_list.append(ie_numpy)
#
#         epe_all = np.concatenate(epe_list)
#         epe = np.mean(epe_all)
#
#         ae_all = np.concatenate(ae_list)
#         ae = np.mean(ae_all)
#
#         ie_all = np.concatenate(ie_list)
#         ie = np.mean(ie_all)
#
#         print("Validation (%s) EPE: %f, AE: %f, IE: %f" % (dstype, epe, ae, ie))
#         results[f"{dstype}_tile"] = np.mean(epe_list)
#
#     return results
#


@torch.no_grad()
def validate_tub(model, sigma=0.05):
    """ Perform validation using the TUBCrowdFlow dataset """

    IMAGE_SIZE = [720, 1280]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    for dstype in ['IM01', 'IM01_hDyn', 'IM02', 'IM02_hDyn', 'IM03', 'IM03_hDyn', 'IM04', 'IM04_hDyn', 'IM05', 'IM05_hDyn']:
        val_dataset = datasets.TubCrowdFlow(dstype=dstype)

        epe_list = []
        ae_list = []
        ie_list = []

        for val_id in tqdm(range(len(val_dataset)), desc=dstype):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            flows = torch.zeros_like(flow_gt, device='cuda').unsqueeze(0)
            flow_count = torch.zeros_like(flow_gt, device='cuda').unsqueeze(0)

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]

                flow_pre, _ = model(image1_tile, image2_tile, flow_init=None)

                padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)

                # 对flow_pre中每个元素进行padding
                flow_pre_padded = [F.pad(flow_pre[i] * weights[idx], padding) for i in range(len(flow_pre))]
                weights_padded = F.pad(weights[idx], padding)

                # 将flow_pre_padded中每个元素加到flows上
                for i in range(len(flow_pre_padded)):
                    flows += flow_pre_padded[i]
                flow_count += weights_padded


            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            # EPE
            epe = torch.sum((flow_pre - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            # AE
            flow_pre_np = flow_pre.permute(1, 2, 0).numpy()
            flow_gt_np = flow_gt.permute(1, 2, 0).numpy()
            ae_np = angular_error(flow_pre_np, flow_gt_np, np.ones_like(flow_gt_np[:, :, 0]))
            ae_list.append(ae_np)

            # IE
            image1_np = image1[0].permute(1, 2, 0).cpu().numpy()
            image2_np = image2[0].permute(1, 2, 0).cpu().numpy()
            warped_image1_np = warp_image_with_flow(image1_np, flow_pre_np)
            ie_np = Interpolation_error(warped_image1_np, image2_np)
            ie_list.append(ie_np)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        ae_all = np.concatenate(ae_list)
        ae = np.mean(ae_all)

        ie_all = np.concatenate(ie_list)
        ie = np.mean(ie_all)

        print("Validation (%s) EPE: %f, AE: %f, IE: %f" % (dstype, epe, ae, ie))
        results[f"{dstype}_tile"] = np.mean(epe_list)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gmflownet', help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--ckpt', default='pretrained_models/gmflownet-things.pth', help="restored checkpoint")
    parser.add_argument('--dataset', default='tub', help="dataset for evaluation")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(create_model(args))
    model.load_state_dict(torch.load(args.ckpt), strict=True)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'sintel_test':
            create_sintel_submission(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'kitti_test':
            create_kitti_submission(model.module)

        elif args.dataset == 'tub':
            validate_tub(model.module)
