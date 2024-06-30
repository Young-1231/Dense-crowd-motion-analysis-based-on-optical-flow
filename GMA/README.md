<div align="center">
  <img src="resources/mmflow-logo.png" width="600"/>
    <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmflow)](https://pypi.org/project/mmflow/)
[![PyPI](https://img.shields.io/pypi/v/mmflow)](https://pypi.org/project/mmflow)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmflow.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmflow/workflows/build/badge.svg)](https://github.com/open-mmlab/mmflow/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmflow/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmflow)
[![license](https://img.shields.io/github/license/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmflow.svg)](https://github.com/open-mmlab/mmflow/issues)

[📘使用文档](https://mmflow.readthedocs.io/en/latest/) |
[🛠️安装教程](https://mmflow.readthedocs.io/en/latest/install.html) |
[👀模型库](https://mmflow.readthedocs.io/en/latest/model_zoo.html) |
[🤔报告问题](https://github.com/open-mmlab/mmflow/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 简介

MMFlow 是一款基于 PyTorch 的光流工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

<https://user-images.githubusercontent.com/76149310/141947796-af4f1e67-60c9-48ed-9dd6-fcd809a7d991.mp4>

### 主要特性

- **首个光流算法的统一框架**

  MMFlow 是第一个提供光流方法统一实现和评估框架的工具箱。

- **模块化设计**

  MMFlow 将光流估计框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的光流算法模型。

- **丰富的开箱即用的算法和数据集**

  MMFlow 支持了众多主流经典的光流算法，例如 FlowNet, PWC-Net, RAFT 等，
  以及多种数据集的准备和构建，如 FlyingChairs, FlyingThings3D, Sintel, KITTI 等。

具体介绍与安装详情请见[这里](https://github.com/open-mmlab/mmflow)

## 本地部署

本项目使用mmflow中的GMA框架在TUBCrowdflow上进行验证，具体部署细节如下:
### 1.下载mmflow算法库到本地

### 2.准备数据集

### 3.准备配置文件

mmflow最终训练和测试以读取的配置文件为准，由于本次调用已有网络模型， 其大概可分为三步：读取注册数据集，针对所调用网络的数据集预处理，组装形成配置文件。配置文件详细介绍可参考官方文档[Tutorial 0: Learn about Configs — mmflow documentation](https://mmflow.readthedocs.io/en/latest/tutorials/0_config.html)

### 4.进行测试

tools/test.py中使用argparse模块从外部导入所需参数，主要有三项，其余视具体需求而定：

'config'：测试所需配置文件路径名

'checkpoint'：预训练模型路径，可提前下载放入项目中

'--eval'：测试指标，如'EPE'

在命令行中加入运行文件名称并输入三个参数，或在Pycharm中为test文件编辑配置避免每次运行重复输入，例如：

`configs/gma/gma_TUB_1280x720_my.py checkpoint/gma_8x2_120k_mixed_368x768.pth --eval EPE`

训练同理，参数具体见代码文件介绍。

### 版本使用

- **CUDA 11.1**
- **Pytorch 1.8.2**
- **mmcv-full 1.6.2**

### 使用权重
- **[gma_8x2_120k_mixed_368x768.pth](checkpoint%2Fgma_8x2_120k_mixed_368x768.pth)**

### 指标结果
|    序列     |  EPE   |   AE   |   IE    |
|:---------:|:------:|:------:|:-------:|
|   IM01    | 0.3626 | 0.1931 | 46.1924 |
| IM01_hDyn | 2.5980 | 1.0754 | 65.0057 |
|   IM02    | 0.2858 | 0.1385 | 41.5246 |
| IM02_hDyn | 2.3496 | 1.0168 | 59.2464 |
|   IM03    | 0.2304 | 0.1583 | 42.5672 |
| IM03_hDyn | 2.4622 | 1.0157 | 67.0546 |
|   IM04    | 0.2121 | 0.1009 | 36.5419 |
| IM04_hDyn | 3.3653 | 1.1477 | 56.8537 |
|   IM05    | 1.1019 | 0.4770 | 74.5594 |
| IM05_hDyn | 6.2401 | 1.1709 | 89.5778 |

### 微调后结果
|    序列     |  EPE   |   AE    |   IE    |
|:---------:|:------:|:-------:|:-------:|
|   IM01    | 0.1788 | 0.0831  | 42.0690 |
|   IM02    | 0.1743 | 0.0733  | 44.6914 |
|   IM03    | 0.1312 | 0.0898  | 41.6692 |
|   IM04    | 0.1603 | 0.0662  | 37.0601 |
|   IM05    | 0.7057 | 0.3525  | 70.8494 |
### 可视化
未微调前IM01

https://github.com/zhengcyyy/Dense-crowd-motion-analysis-based-on-optical-flow/assets/150784185/de58f18e-f7a8-47af-a211-20e621e78b4f

微调后IM01

https://github.com/zhengcyyy/Dense-crowd-motion-analysis-based-on-optical-flow/assets/150784185/11dcfbb7-f74d-4bb0-855f-e660e8c73bad

### 注意

❗ **mmflow/models/flow_estimators/raft.py** **forward
_train**中取消了valid标记

❗ **mmflow/core/evaluation/metrics.py**中添加了AE、IE指标
