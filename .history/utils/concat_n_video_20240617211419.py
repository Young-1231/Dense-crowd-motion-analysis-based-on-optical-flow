import cv2
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate a video from a folder of images"
    )
    parser.add_argument(
        "--video1",
        type=str,
        default="output/video/1.mp4",
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--video2",
        type=str,
        default="output/video/2.mp4",
        help="Path to the output video file",
    )
    parser.add_argument(
        "output",
        type=str,
        default="3.mp4",
        help="Path to the output video file",
    )

    return parser


def main(args):
    if not os.path.exists("output/video/"):
        os.makedirs("output/video/")

    # 输入视频文件路径
    video1_path = args.video1
    video2_path = args.video2
    # 输出视频文件路径
    output_video_path = "output/video/" + args.output
    # 视频帧率（假设两个视频帧率相同）
    frame_rate = 20.0

    # 打开视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取视频的帧宽度和高度
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确定输出视频的宽度和高度
    output_width = width1 + width2
    output_height = max(height1, height2)

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用'mp4v'编码器生成MP4文件
    out = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (output_width, output_height)
    )

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果任意一个视频结束，则停止
        if not ret1 or not ret2:
            break

        # 如果视频尺寸不一样，需要调整
        if height1 != height2:
            frame1 = cv2.resize(frame1, (width1, output_height))
            frame2 = cv2.resize(frame2, (width2, output_height))

        # 拼接两帧图像
        combined_frame = cv2.hconcat([frame1, frame2])

        # 写入输出视频
        out.write(combined_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()

    print(f"视频已成功保存到 {output_video_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
