import sys
sys.path.append('core')

from PIL import Image
import argparse
import torch
from core import create_model
from thop import profile, clever_format

TRAIN_SIZE = [432, 960]

def build_model(args):
    model = create_model(args)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gmflownet', help="model class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--ckpt', default='checkpoints/gmflownet-things.pth', help="restored checkpoint")
    parser.add_argument('--dataset', default='tub', help="dataset for evaluation")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(args)
    model.to(device)

    # Wrap model with DataParallel after moving it to the device
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    image1 = torch.rand(1, 3, 432, 960).to(device)
    image2 = torch.rand(1, 3, 432, 960).to(device)

    # Ensure all parameters are on the correct device
    for param in model.parameters():
        assert param.device == device

    macs, params = profile(model, inputs=(image1, image2))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)
