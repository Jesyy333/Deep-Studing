# 数据集说明 PeMS Highway Traffic Dataset
本项目使用加州高速路网公开时序交通数据集：METR-LA、PeMS-BAY，用于车流时空推演模型训练。

## 1. 数据集下载地址
官方开源仓库：https://github.com/liyaguang/DCRNN
项目仓库中 `data/` 文件夹可直接下载完整 h5 数据与距离矩阵 `.npy` 文件。

备用镜像下载（国内加速）：
https://kkgithub.com/liyaguang/DCRNN

## 2. 文件存放路径要求
下载完成后，将以下文件全部放入本 `data/` 目录：
- METR-LA.h5
- METR-LA_dist.npy
- PeMS-BAY.h5
- PeMS-BAY_dist.npy

最终目录结构：
