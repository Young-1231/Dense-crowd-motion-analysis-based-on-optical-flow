import json
import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
from tqdm import tqdm
import pickle


def create_mask(image_shape, polygon):
    """
    Create a binary mask with 1 inside the polygon and 0 outside.

    :param image_shape: tuple of (height, width) for the output mask
    :param polygon: list of (x, y) tuples representing the polygon vertices
    :return: numpy array of shape (height, width) with 1 inside the polygon and 0 outside
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create a grid of coordinates (x, y)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coordinates = np.stack((x, y), axis=-1).reshape(-1, 2)

    # Check which coordinates are inside the polygon
    poly_path = mplPath.Path(polygon)
    inside = poly_path.contains_points(coordinates)

    # Reshape the mask
    mask = inside.reshape((height, width)).astype(np.uint8)

    return mask


def convert(json_file):

    with open(json_file, "r") as f:
        data = json.load(f)

    shapes = data["shapes"]
    picture_name = data["imagePath"]

    direction = np.zeros((data["imageHeight"], data["imageWidth"], 2), dtype=np.int8)
    id = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.int8)

    num_shapes = len(shapes)

    shape_id = 0
    for shape in shapes:
        shape_id += 1

        polygon = shape["points"]
        mask = create_mask((data["imageHeight"], data["imageWidth"]), polygon)

        # expand
        mask_bool = mask.astype(bool)
        format_mask = np.zeros_like(direction)

        if shape["label"] == "up":
            flag = np.array([0, 1])
        elif shape["label"] == "down":
            flag = np.array([0, -1])
        elif shape["label"] == "left":
            flag = np.array([1, 0])
        elif shape["label"] == "right":
            flag = np.array([-1, 0])
        else:
            raise ValueError(f"Unknown label: {shape['label']}")

        format_mask[mask_bool] = flag
        id[mask_bool] = shape_id

        direction += format_mask

    return direction, num_shapes, id


class Label:
    def __init__(self, direction, num_shapes, id, picture_name):
        self.direction = direction
        self.num_shapes = num_shapes
        self.id = id
        self.shape = direction.shape
        self.picture_name = picture_name


def pack_label(json_file):
    direction, num_shapes, id = convert(json_file)
    picture_name = os.path.basename(json_file).replace(".json", ".jpg")
    label = Label(direction, num_shapes, id, picture_name)
    return label


def main(args):
    files = glob(os.path.join(args.folder, "*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {args.folder}")

    print("转换开始...")

    for file in tqdm(files):
        mask = convert(file)
        # print(mask)
        # plt.imshow(mask[:, :, 1])
        # plt.show()
        np.save(file.replace(".json", ".npy"), mask)

    print("转换完成！")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="E:/E盘/模式识别课设/jhroad_label",
        help="Path to the JSON file",
    )
    args = parser.parse_args()

    main(args)
