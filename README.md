# Dense-crowd-motion-analysis-based-on-optical-flow

## 简介

本项目基于mmflow首先实现在TUB CrowdFlow数据集上对比例如 RAFT、FlowNet2.0等主流光流模型，并在该数据集上报告EPE、AE、IE等光流评价指标的对比结果。

## mmflow本地部署

### 1.下载mmflow算法库到本地

### 2.准备数据集

### 3.准备配置文件

mmflow最终训练和测试以读取的配置文件为准，由于本次调用已有网络模型，其大概可分为三步：读取数据集，针对所调用网络的数据集预处理，组装形成配置文件。配置文件详细介绍可参考官方文档[Tutorial 0: Learn about Configs — mmflow documentation](https://mmflow.readthedocs.io/en/latest/tutorials/0_config.html)

### 4.进行测试

tools/test.py中使用argparse模块从外部导入所需参数，主要有三项，其余视具体需求而定：

'config'：测试所需配置文件路径名

'checkpoint'：预训练模型路径，可提前下载放入项目中

'--eval'：测试指标，如'EPE'

在命令行中输入三个参数，或在Pycharm中为test文件编辑配置避免每次运行重复输入，例如：

`configs/gma/gma_TUB_1280x720_my.py checkpoint/gma_8x2_120k_mixed_368x768.pth --eval EPE`
