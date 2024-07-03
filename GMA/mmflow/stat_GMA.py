####
# 基于GMA原始代码库的估计
# --model checkpoints/gma-sintel.pth --dataset sintel
####

import sys

import argparse

import torch

from core.network import RAFTGMA

from thop import profile, clever_format

sys.path.append('core')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.DataParallel(RAFTGMA(args))
    # 由于DataParallel抱起来model，所以下面profile必须要.module
    model.load_state_dict(torch.load(args.model))

    model.to('cuda:0')

    model.eval()

    image1 = torch.rand(1, 3, 432, 960).to('cuda:0')
    image2 = torch.rand(1, 3, 432, 960).to('cuda:0')

    macs, params = profile(model.module, inputs=(image1, image2))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)