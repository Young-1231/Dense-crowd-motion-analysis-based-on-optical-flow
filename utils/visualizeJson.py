import argparse
import os

import os.path as osp

import json
import base64

import imgviz
import PIL.Image

from labelme import utils

import cv2
from tqdm import tqdm


def get_args_parser():
    _parser = argparse.ArgumentParser(
        description="Convert json labels to mp4 video",
    )
    _parser.add_argument(
        "root_path",
        type=str,
        help="Root paths of json labels",
    )
    return _parser


def visualize_image(data_dir):
    # read json files
    suffixs = ['.json']
    jsons = None
    for suffix in suffixs:
        jsons = [file for file in os.listdir(data_dir) if file.endswith(suffix)]
        if len(jsons) > 0:
            break
    jsons.sort()

    # create output directories
    if not os.path.exists(osp.join(data_dir, 'label')):
        os.makedirs(osp.join(data_dir, 'label'))
    if not os.path.exists(osp.join(data_dir, 'label_viz')):
        os.makedirs(osp.join(data_dir, 'label_viz'))

    # output images
    for _json in tqdm(jsons, desc='Visualizing images ...'):
        json_file = os.path.join(data_dir, _json)

        data = json.load(open(json_file))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )

        out_name = _json.replace(".json", ".png")

        # original image
        # PIL.Image.fromarray(img).save(osp.join(data_dir, 'ori', out_name)) # no need

        # label image
        utils.lblsave(osp.join(data_dir, 'label', out_name), lbl)

        # original image with labels
        PIL.Image.fromarray(lbl_viz).save(osp.join(data_dir, 'label_viz', out_name))

    return osp.join(data_dir, 'label'), osp.join(data_dir, 'label_viz')


def visualize_video(data_dir, out_name, frame_rate=5.0):
    # create output directory
    out_dir = os.path.join(data_dir, 'video')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read images
    images = None
    suffixs = ['.png', '.jpg', '.jpeg']
    for suffix in suffixs:
        images = [file for file in os.listdir(data_dir) if file.endswith(suffix)]
        if len(images) > 0:
            break
    images.sort()
    first_image_path = os.path.join(data_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    print(f"height: {height}, width: {width}, layers: {layers}")

    # output flow video
    video_file = os.path.join(out_dir, out_name)
    frame_rate = frame_rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))
    for image in tqdm(images, desc='Creating video ...'):
        image_path = os.path.join(data_dir, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {video_file}")


def main(_args):
    root_path = _args.root_path
    label_root_path, label_viz_root_path = visualize_image(root_path)
    visualize_video(label_root_path, out_name='label_video.mp4')
    visualize_video(label_viz_root_path, out_name='label_viz_video.mp4')


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
