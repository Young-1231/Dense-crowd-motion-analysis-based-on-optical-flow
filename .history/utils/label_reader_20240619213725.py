import json
import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt

json_file = "E:/E盘/模式识别课设/label/frame_0001.json"

with open(json_file, "r") as f:
    data = json.load(f)


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


# 示例顶点 (根据实际使用情况替换)
polygon = [(100, 200), (150, 400), (300, 350), (250, 150)]

# 创建一个大小为(500, 500)的图像的掩码
image_shape = (500, 500)
mask = create_mask(image_shape, polygon)

# 可视化结果
plt.imshow(mask, cmap="gray")
plt.show()
