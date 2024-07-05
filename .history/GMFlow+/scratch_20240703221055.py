from utils.flow_viz import save_vis_flow_tofile
import numpy as np


def load_flow_to_numpy(path):
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


data = load_flow_to_numpy("G:/dataset/TUBCrowdFlow/gt_flow/IM01/frameGT_0000.flo")
save_vis_flow_tofile(data, "G:/dataset/TUBCrowdFlow/gt_flow/IM01/frameGT_0000_new.png")
