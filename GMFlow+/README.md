# 基于CrowdFlow数据集上的GMFlow+模型测试
本项目基于[Unifying Flow, Stereo and Depth Estimation](https://github.com/autonomousvision/unimatch)项目代码开发，加入了[CrowdFlow](https://github.com/tsenst/CrowdFlow)数据集并进行测试。

## 环境配置
使用conda：
```
conda env create -f conda_environment.yml
conda activate unimatch
```
使用pip：
```
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install imageio==2.9.0 imageio-ffmpeg matplotlib opencv-python pillow scikit-image scipy tensorboard==2.9.1 setuptools==59.5.0
```

## 数据集
请将[CrowdFlow](https://github.com/tsenst/CrowdFlow)数据集放在GMFlow+目录下的datasets文件夹，或者在`--tub_root`参数中传入自定义的文件夹。

## 运行用例
```
python main_flow.py --eval --resume pretrained/gmflow-scale1-things-e9887eda.pth --val_dataset tub --with_speed_metric --tub_root datasets/TUBCrowdFlow --tub_IM 1
```

## 测试结果
|    序列     |  EPE   |   AE   |   IE    |
|:---------:|:------:|:------:|:-------:|
| IM01 | 0.53554 | 0.23989 | 32.97178|
| IM01_hDyn | 0.36959 | 0.2218 | 27.59227|
| IM02 | 0.43167 | 0.25416 | 32.64068|
| IM02_hDyn | 0.30441 | 0.19083 | 23.70604|
| IM03 | 1.78008 | 0.69063 | 48.85698|
| IM03_hDyn | 4.02510 | 1.23368 | 41.97871|
| IM04 | 3.76057 | 1.20666 | 41.45065|
| IM04_hDyn | 4.09951 | 1.23033 | 47.98547|
| IM05 | 4.06573 | 1.26431 | 36.34635|
| IM05_hDyn | 6.64464 | 1.23461 | 54.78697|