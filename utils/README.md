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
python utils/ts2png.py E:/data/Wuhan_Metro/江汉路-厅东闸机2-2-20231231170000-20231231203000-29254227/
```
