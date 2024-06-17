import subprocess
import os


def extract_frames(_video_path, _output_dir, duration=400, fps=1):
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


root_path = 'E:/data/Wuhan_Metro/'
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.ts'):
            video_path = os.path.join(root, file)
            output_dir = os.path.join(root, file.split('.')[0])
            print('Extracting frames from', video_path, 'to', output_dir, '...')
            extract_frames(video_path, output_dir)
