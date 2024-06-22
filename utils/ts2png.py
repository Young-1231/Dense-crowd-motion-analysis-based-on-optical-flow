import argparse
import subprocess
import os


def get_args_parser():
    _parser = argparse.ArgumentParser(
        description="Convert ts video to png frames",
    )
    _parser.add_argument(
        "root_path",
        type=str,
        help="Root paths of videos",
    )
    return _parser


def extract_frames(_video_path, _output_dir, duration=30, fps=25):
    os.makedirs(_output_dir, exist_ok=True)

    output_pattern = os.path.join(_output_dir, 'frame_%04d.png')
    command = [
        'ffmpeg',
        '-i', _video_path,
        '-t', str(duration),
        '-vf', f'fps={fps}',
        output_pattern
    ]

    subprocess.run(command, check=True)


def main(_args):
    root_path = _args.root_path
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.ts'):
                video_path = os.path.join(root, file)
                output_dir = os.path.join(root, file.split('.')[0])
                print('Extracting frames from', video_path, 'to', output_dir, '...')
                extract_frames(video_path, output_dir)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)


