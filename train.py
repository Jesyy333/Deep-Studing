import torch
import torch.nn as nn
import torch.optim as optim
from data.load_pems import load_pems_dataset
from models.dcrnn import DCRNN
from models.stgformer import STGformer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="dcrnn", choices=["dcrnn","stgformer"])
parser.add_argument("--dataset", type=str, default="METR-LA")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--pred_len", type=int, default=12)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据
data, adj, _ = load_pems_dataset("./data", args.dataset)
T, N, C = data.shape
split1, split2 = int(T*0.7), int(T*0.8)
train_data = torch.FloatTensor(data[:split1]).to(device)
val_data = torch.FloatTensor(data[split1:split2]).to(device)
adj = torch.FloatTensor(adj).to(device)

# 构造时序滑窗样本
def create_seq(data, hist=12, pred=12):
    xs, ys = [], []
    for i in range(len(data)-hist-pred):
        xs.append(data[i:i+hist])
        ys.append(data[i+hist:i+hist+pred, :, 0:1])
    return torch.stack(xs), torch.stack(ys)

x_train, y_train = create_seq(train_data)
x_val, y_val = create_seq(val_data)

# 初始化模型
if args.model == "dcrnn":
    model = DCRNN(adj, in_dim=C, pred_len=args.pred_len).to(device)
elif args.model == "stgformer":
    model = STGformer(adj, in_dim=C, pred_len=args.pred_len).to(device)

loss_fn = nn.L1Loss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for i in tqdm(range(0, len(x_train), 32)):
        xb = x_train[i:i+32]
        yb = y_train[i:i+32]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = loss_fn(val_pred, y_val)
    print(f"Epoch{epoch:3d} TrainLoss:{total_loss/len(x_train):.4f} ValLoss:{val_loss:.4f}")
# 保存权重（边缘推理使用）
torch.save(model.state_dict(), f"./ckpt/{args.model}_{args.dataset}.pth")
