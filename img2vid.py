import cv2
import os
import re

# 设置输入文件夹路径和输出视频文件路径
input_folder = '/home/guest/github/hold/generator/hamer/demo_out'  # 替换成你实际的文件夹路径
output_video = '/home/guest/github/hold/generator/hamer/output_video.mp4'  # 输出的视频文件

# 获取文件夹内的所有符合 xxxx_all.jpg 格式的图片
image_files = [f for f in os.listdir(input_folder) if re.match(r'\d{4}_all\.jpg', f)]

# 按照文件名中的四位数字进行排序
image_files.sort(key=lambda x: int(re.match(r'(\d{4})_all\.jpg', x).group(1)))

# 读取第一张图片来获取视频的尺寸
first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

# 设置视频写入对象，使用mp4格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 你可以使用 'XVID' 或其他格式
video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

# 遍历图片并将它们写入视频
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)
    video_writer.write(img)

# 释放视频写入对象
video_writer.release()

print(f"视频已保存到 {output_video}")
