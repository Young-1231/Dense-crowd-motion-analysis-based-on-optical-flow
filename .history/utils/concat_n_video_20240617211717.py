import cv2
import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Generate a video from multiple input videos"
    )
    parser.add_argument(
        "videos",
        type=str,
        nargs="+",
        help="Paths to the input video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/video/output.mp4",
        help="Path to the output video file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate of the output video",
    )
    return parser


def main(args):
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # 输入视频文件路径
    video_paths = args.videos
    # 输出视频文件路径
    output_video_path = args.output
    # 视频帧率（假设所有视频帧率相同）
    frame_rate = args.fps

    # 打开所有视频文件
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    # 获取所有视频的帧宽度和高度
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

    # 确定输出视频的宽度和高度
    output_width = sum(widths)
    output_height = max(heights)

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用'mp4v'编码器生成MP4文件
    out = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (output_width, output_height)
    )

    while True:
        rets = [cap.read() for cap in caps]
        if not all(ret[0] for ret in rets):
            break

        frames = [ret[1] for ret in rets]

        # 调整所有帧的高度一致
        resized_frames = [
            (
                cv2.resize(frame, (width, output_height))
                if frame.shape[0] != output_height
                else frame
            )
            for frame, width in zip(frames, widths)
        ]

        # 拼接所有帧
        combined_frame = cv2.hconcat(resized_frames)

        # 写入输出视频
        out.write(combined_frame)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()

    print(f"视频已成功保存到 {output_video_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
