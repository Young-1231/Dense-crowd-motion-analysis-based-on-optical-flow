import cv2
import os

# 图片文件夹路径
image_folder = 'E:\mmflow\TUBCrowdFlow\TUBCrowdFlow\gt_flow\IM01' #'images/IM01finetune'
# 视频文件输出路径
video_file = 'gt.mp4'
# 视频帧率
frame_rate = 20.0

# 获取所有图片文件
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # 确保图片按顺序排列

# 获取第一张图片的尺寸
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 定义视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用'mp4v'编码器生成MP4文件
video = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))

# 遍历所有图片并写入视频
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)

# 释放视频写入对象
video.release()

print(f"视频已成功保存到 {video_file}")