# Dense-crowd-motion-analysis-based-on-optical-flow

## 简介

- 在 TUB CrowdFlow 数据集上对比主流光流模型，并在该数据集上报告 EPE、AE、IE 等光流评价指标的对比结果；
- 基于一个合适的光流模型，在 WuhanMetro 数据集上进行人群运动分析，实现:
  - 人群运动方向的识别，要求获取不同人群运动方向的光流向量，输出不同人群运动
方向的数量并按上图的形式区分显示，以 MAE 和 MSE 为评测指标；
  - 人群运动区域分割，获取不同运动方向人群的分割图并评估分割区域的 Pixel 
Accuracy 和 mIOU 等指标；
- 集成所选算法，在 WuhanMetro 数据集选取典型场景完成 demo 的制作，输出光流、
人群运动场、人群分割图等处理结果。

## 调用模型

- **Ef-RAFT(CVPR 2024)**
- **Flow-Former(ECCV 2022)**
- **GMA(ICCV 2021)**
- **GMFlow+(TPAMI 2023)**
- **GMFlowNet(CVPR 2022)**

### 测试结果

各模型测试结果详解项目子文件。可视化结果如下

### 参考文献

* Baker, S., Scharstein, D., Lewis, J.P. et al. A Database and Evaluation Methodology for Optical Flow. Int J Comput Vis 92, 1–31 (2011). https://doi.org/10.1007/s11263-010-0390-2