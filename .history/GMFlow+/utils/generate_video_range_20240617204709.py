import cv2
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate a video from a folder of images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./images/IM01finetune",
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="output.mp4",
        help="Path to the output video file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate of the output video",
    )
    parser.add_argument(
        "--range",
        type=str,
        default="all",
        help="Range of images to include in the video. Options: 'all', '%d-%d'",
    )

    return parser


def main(args):
    if not os.path.exists("output/video/"):
        os.makedirs("output/video/")

    # 图片文件夹路径
    image_folder = args.image_folder
    # 视频文件输出路径
    video_file = "output/video/" + args.video_name
    # 视频帧率
    frame_rate = args.fps

    # 获取所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 确保图片按顺序排列

    if args.range != "all":
        start, end = map(int, args.range.split("-"))
        images = images[start:end]

    # 获取第一张图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用'mp4v'编码器生成MP4文件
    video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))

    # 遍历所有图片并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()

    print(f"视频已成功保存到 {video_file}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
