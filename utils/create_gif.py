import argparse
import imageio.v2 as imageio
import os
from skimage.transform import resize
import numpy as np

def extract_number(filename):
    base_name = os.path.basename(filename)
    number_part = base_name.split('_')[1].split('.')[0]
    return int(number_part)

def main(subfolder, target_duration, target_fps, new_width, new_height):
    image_paths = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.startswith('frame_') and f.endswith('.png')]
    image_paths.sort(key=extract_number)
    total_frames = int(target_duration * target_fps)
    start_frame_idx = 0
    end_frame_idx = min(10, len(image_paths) - 1)
    step_size = (end_frame_idx - start_frame_idx + 1) / total_frames
    frame_indices = [int(start_frame_idx + i * step_size) % (end_frame_idx + 1) for i in range(total_frames)]

    with imageio.get_writer('whu_1_3_loop.gif', mode='I', fps=target_fps, loop=0) as writer:
        for idx in frame_indices:
            path = image_paths[idx]
            image = imageio.imread(path)
            image_resized = resize(image, (new_height, new_width), anti_aliasing=True)
            writer.append_data((image_resized * 255).astype(np.uint8))

    print("Custom duration looping GIF animation created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create custom duration looping GIF animation from PNG frames.')
    parser.add_argument('subfolder', type=str, help='Path to the folder containing PNG frames')
    parser.add_argument('--duration', type=int, default=1, help='Target duration of the GIF animation in seconds')
    parser.add_argument('--fps', type=int, default=10, help='Target frames per second (fps) of the GIF animation')
    parser.add_argument('--width', type=int, default=426, help='Width of the resized frames')
    parser.add_argument('--height', type=int, default=240, help='Height of the resized frames')
    args = parser.parse_args()

    main(args.subfolder, args.duration, args.fps, args.width, args.height)
