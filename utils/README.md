# 小工具

## concat_n_video.py
从多个输入视频生成一个视频。

用法示例：
```sh
python utils/concat_n_video.py video1.mp4 video2.mp4 video3.mp4 --output output.mp4 --fps 25 
```

## label_reader.py
将labelme的标注数据直接转换为npy数组。

用法示例：
```sh
python utils/label_reader.py {JSON_FOLDER}
```

## ts2png.py
将ts文件转换为png图片。

用法示例：
```sh
python utils/ts2png.py E:/data/Wuhan_Metro/
```

## visualizeJson.py
将labelme的标注数据可视化，输出每帧图片和视频。

用法示例：
```sh
python utils/visualizeJson.py E:/data/Wuhan_Metro/江汉路-厅东闸机2-2-20231231170000-20231231203000-29254227/
```

# 标签
用于处理与分析标签的代码。

## 数据结构


代码可以从labelme的json文件转化为直接使用的numpy数组，用于表示运动方向。
文件输出为`{FILE_NAME}.pkl`存储的字典文件。
\
`data`：数组大小`(H, W, 2)`，其中第三维为2维向量，一般采用四方向量化。即`[0,1]`表示`up`，`[0,-1]`表示`down`等。
\
`num_shapes`：运动区域数量。
\
`id`：数组大小`(H, W)`，指示像素对应的运动区域编号，0为无运动。
\
`shape`：图像大小。
\
`picture_name`：原始图像名称。

## 调用方法
```python
import pickle

with open("frame_0003.pkl","rb") as f:
    label = pickle.load(f)

print(label["data"])
print(label["num_shapes"])
print(label["id"])
print(label["shape"])
print(label["picture_name"])

```

## Label Reader
默认生成单独的文件，传入参数`--pack True`可以生成单个pkl文件。pkl文件保存了一个label列表。