import numpy as np
import pandas as pd
import os

def gaussian_kernel(dist_matrix, sigma=10):
    """构建路网高斯加权邻接矩阵，匹配加州高速拥堵扩散机制"""
    dist = dist_matrix.copy()
    dist[dist == 0] = np.inf
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    adj[adj < 0.01] = 0
    return adj.astype(np.float32)

def load_pems_dataset(data_root, dataset="METR-LA"):
    """
    加载加州PeMS基准数据集：METR-LA / PeMS-BAY / PeMSD7
    返回: traffic_data(时间×传感器×特征), adj_matrix, timestamp
    """
    data_path = os.path.join(data_root, f"{dataset}.h5")
    dist_path = os.path.join(data_root, f"{dataset}_dist.npy")

    df = pd.read_hdf(data_path)
    dist_matrix = np.load(dist_path)

    # 时间戳处理，提取日/周周期特征
    timestamps = pd.to_datetime(df.index)
    time_feat = np.stack([
        timestamps.hour / 24,
        timestamps.dayofweek / 7
    ], axis=-1)

    traffic = df.values.astype(np.float32)  # [T, N]
    T, N = traffic.shape

    # 缺失值基础填充（ffill，对应综述Ghazi混合网络前置处理）
    traffic = pd.DataFrame(traffic).fillna(method="ffill").fillna(0).values

    # 拼接周期时间特征
    data = np.concatenate([traffic[..., None], np.tile(time_feat[:, None, :], (1, N, 1))], axis=-1)

    adj = gaussian_kernel(dist_matrix)
    return data, adj, timestamps

if __name__ == "__main__":
    data, adj, ts = load_pems_dataset("./data", "METR-LA")
    print(f"数据集形状: {data.shape}, 传感器数量: {adj.shape[0]}")
