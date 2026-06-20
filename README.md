# 面向加州高速路网车流推演的时空时序模型及边缘工程落地
复现论文：面向加州高速路网车流推演的时空时序模型及边缘工程落地研究进展
数据集：PeMS (METR-LA / PeMS-BAY / PeMSD7)
实现模型：
1. DCRNN：扩散卷积循环网络[1] 时序基线模型
2. STGCN：纯卷积时空图网络[2]
3. Traffic Transformer：动态时空注意力[5]
4. STGformer：单层轻量化注意力，边缘部署专用[8]
5. TL-DCRNN：跨区域迁移学习[3]

## 功能对应综述章节
1. data/load_pems.py：PeMS数据加载、缺失值填充、路网邻接矩阵构建（2.2缺失鲁棒建模）
2. models/：四大主流时空预测模型复现（2.1模型性能对比）
3. train.py：统一训练、METR-LA/湾区数据集对比实验
4. edge_infer.py：模型量化、低算力边缘设备推理（3.3边缘实时性平衡）
5. transfer_dcrnn.py：LA-SF跨域迁移学习（3.4部署成本控制）

## 快速运行
1. 安装依赖
pip install -r requirements.txt
2. 训练DCRNN基线模型
python train.py --model dcrnn --dataset METR-LA
3. 轻量化边缘模型训练
python train.py --model stgformer --dataset PeMS-BAY
4. 边缘设备推理测速
python edge_infer.py
