import torch
import time
from models.stgformer import STGformer
from data.load_pems import load_pems_dataset

device = torch.device("cpu") # 模拟边缘无GPU场景
data, adj, _ = load_pems_dataset("./data", "METR-LA")
adj = torch.FloatTensor(adj).to(device)

# 加载轻量化STGformer
model = STGformer(adj).to(device)
ckpt = torch.load("./ckpt/stgformer_METR-LA.pth", map_location=device)
model.load_state_dict(ckpt)
model.eval()

# 模拟单条实时车流输入（RSU传感器5min数据）
sample = torch.FloatTensor(data[0:12]).unsqueeze(0).to(device)

# 计时推理延迟
start = time.time()
with torch.no_grad():
    res = model(sample)
cost = (time.time() - start)*1000
print(f"边缘推理耗时: {cost:.2f} ms")
print(f"预测车流形状: {res.shape}")

# 模型量化压缩（进一步降低边缘内存占用）
quant_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quant_model.state_dict(), "./ckpt/stgformer_quant_edge.pth")
print("量化轻量化模型已导出，适配高速RSU边缘设备")
