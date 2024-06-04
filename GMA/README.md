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

### 版本使用

- **CUDA 11.1**
- **Pytorch 1.8.2**
- **mmcv-full 1.6.2**

### 指标结果
|    序列     |  EPE   |   AE   |   IE    |
|:---------:|:------:|:------:|:-------:|
|   IM01    | 0.3626 | 0.1931 | 46.1924 |
| IM01_hDyn | 0.2858 | 0.1385 | 41.5246 |
|   IM02    | 0.2304 | 0.1583 | 42.5672 |
| IM02_hDyn | 0.2121 | 0.1009 | 36.5419 |
|   IM03    | 1.1019 | 0.4770 | 74.5594 |
| IM03_hDyn | 2.5980 | 1.0754 | 65.0057 |
|   IM04    | 2.3496 | 1.0168 | 59.2464 |
| IM04_hDyn | 2.4622 | 1.0157 | 67.0546 |
|   IM05    | 3.3653 | 1.1477 | 56.8537 |
| IM05_hDyn | 6.2401 | 1.1709 | 89.5778 |
