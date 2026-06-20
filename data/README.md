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

## 3. 数据简要介绍
1. METR-LA：洛杉矶高速路网，207个传感器，采集时长4个月；
2. PeMS-BAY：旧金山湾区高速路网，325个传感器，采集时长6个月；
数据包含5分钟粒度车流量、平均车速、占有率三类交通特征。

## 4. 运行注意事项
1. 数据集文件体积较大，不随Git仓库上传，需使用者自行下载放置；
2. 运行 `train.py` 时会自动读取本目录下对应数据集文件；
3. 缺失文件会直接触发路径报错，请检查文件命名与存放位置。
