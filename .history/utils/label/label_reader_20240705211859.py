import json  # 导入JSON模块，用于处理JSON数据
import numpy as np  # 导入NumPy模块，用于处理数组
import matplotlib.path as mplPath  # 导入matplotlib.path模块，用于创建多边形路径
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import argparse  # 导入argparse模块，用于解析命令行参数
import os  # 导入os模块，用于操作文件和目录
from glob import glob  # 从glob模块导入glob函数，用于查找符合特定模式的文件
from tqdm import tqdm  # 从tqdm模块导入tqdm函数，用于显示进度条
import pickle  # 导入pickle模块，用于序列化和反序列化数据

"""
标签转换工具

将文件夹下LabelMe标注的json文件转换为pkl文件，将点记录转化为像素级别的标签，与光流矩阵形式统一。
数据结构见README文件。
"""


def create_mask(image_shape, polygon):
    """
    创建二进制掩码，多边形内为 1，多边形外为 0。

    :param image_shape: 输出掩码的（高度、宽度）元组
    :param polygon: 表示多边形顶点的（x，y）元组列表
    :return: 形状（高度、宽度）的 numpy 数组，多边形内部为 1，外部为 0
    """
    height, width = image_shape  # 获取掩码的高度和宽度
    mask = np.zeros((height, width), dtype=np.uint8)  # 初始化掩码数组，全为0

    # 创建坐标网格 (x, y)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coordinates = np.stack((x, y), axis=-1).reshape(-1, 2)  # 将网格坐标展平

    # 检查哪些坐标点在多边形内
    poly_path = mplPath.Path(polygon)  # 创建多边形路径
    inside = poly_path.contains_points(coordinates)  # 检查每个坐标点是否在多边形内

    # 重塑掩码
    mask = inside.reshape((height, width)).astype(
        np.uint8
    )  # 将布尔值转换为0或1并重塑掩码

    return mask


def convert(json_file):
    """
    转换单个JSON文件中的标注数据。

    :param json_file: JSON文件的路径
    :return: direction数组、形状数量、id数组和图片名称
    """
    with open(json_file, "r") as f:
        data = json.load(f)  # 读取JSON文件内容

    shapes = data["shapes"]  # 获取标注形状信息
    picture_name = data["imagePath"]  # 获取图片名称

    direction = np.zeros(
        (data["imageHeight"], data["imageWidth"], 2), dtype=np.int8
    )  # 初始化方向矩阵
    id = np.zeros(
        (data["imageHeight"], data["imageWidth"]), dtype=np.int8
    )  # 初始化id矩阵

    num_shapes = len(shapes)  # 获取形状数量

    shape_id = 0
    for shape in shapes:
        shape_id += 1

        polygon = shape["points"]  # 获取形状的多边形顶点
        mask = create_mask(
            (data["imageHeight"], data["imageWidth"]), polygon
        )  # 创建掩码

        # 扩展
        mask_bool = mask.astype(bool)  # 将掩码转换为布尔类型
        format_mask = np.zeros_like(direction)  # 创建与direction相同形状的格式化掩码

        if shape["label"] == "top" or shape["label"] == "up":
            flag = np.array([0, 1])  # 设置上方向标志
        elif shape["label"] == "down":
            flag = np.array([0, -1])  # 设置下方向标志
        elif shape["label"] == "left":
            flag = np.array([1, 0])  # 设置左方向标志
        elif shape["label"] == "right":
            flag = np.array([-1, 0])  # 设置右方向标志
        else:
            raise ValueError(f"Unknown label: {shape['label']}")  # 抛出未知标签异常

        format_mask[mask_bool] = flag  # 在掩码中设置方向标志
        id[mask_bool] = shape_id  # 在掩码中设置形状id

        direction += format_mask  # 累加方向

    return (
        direction,
        num_shapes,
        id,
        picture_name,
    )  # 返回方向矩阵、形状数量、id矩阵和图片名称


def pack_label(json_file):
    """
    将转换后的标注数据打包成一个字典。

    :param json_file: JSON文件的路径
    :return: 包含标注数据的字典
    """
    direction, num_shapes, id, picture_name = convert(json_file)  # 调用convert函数
    label = {}
    label["data"] = direction  # 存储方向数据
    label["num_shapes"] = num_shapes  # 存储形状数量
    label["id"] = id  # 存储id数据
    label["shape"] = direction.shape  # 存储方向矩阵的形状
    label["picture_name"] = picture_name  # 存储图片名称

    return label  # 返回标注字典


def main(args):
    """
    主函数，处理命令行参数并转换文件夹中的所有JSON文件。

    :param args: 命令行参数
    """
    files = glob(os.path.join(args.folder, "*.json"))  # 获取文件夹中所有JSON文件
    if not files:
        raise ValueError(f"No JSON files found in {args.folder}")  # 抛出未找到文件异常

    print("转换开始...")
    label_list = []

    for file in tqdm(files):
        label = pack_label(file)  # 调用pack_label函数
        if args.pack:
            label_list.append(label)  # 将标注添加到列表
        else:
            with open(file.replace(".json", ".pkl"), "wb") as f:
                pickle.dump(label, f)  # 将标注保存为pkl文件

    if args.pack:
        path = args.folder
        _, tail = os.path.split(path)  # 获取文件夹名称
        with open(os.path.join(args.folder, f"{tail}.pkl"), "wb") as f:
            pickle.dump(label_list, f)  # 将所有标注打包保存为一个pkl文件

    print("转换完成！")


if __name__ == "__main__":
    """
    程序入口，解析命令行参数并调用主函数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="E:/E盘/模式识别课设/jhroad_label",
        help="JSON文件所在的文件夹路径",
    )
    parser.add_argument(
        "--pack",
        type=bool,
        default=False,
        help="是否打包为一个pkl文件",
    )

    args = parser.parse_args()

    main(args)  # 调用主函数
