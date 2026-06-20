import torch
import torch.nn as nn

class STGAttentionBlock(nn.Module):
    """单层时空图注意力，大幅降低推理开销，适配RSU边缘设备"""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.head = heads
        self.qkv = nn.Linear(dim, dim*3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, adj):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head, C//self.head).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) / (C**0.5)
        # 引入路网邻接约束，减少无效注意力计算
        attn = attn + adj[None, None, :, :] * -1e9
        attn = F.softmax(attn, -1)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.out(x)

class STGformer(nn.Module):
    def __init__(self, adj, in_dim=3, hidden=32, pred_len=12):
        super().__init__()
        self.N = adj.shape[0]
        self.adj = adj
        self.embed = nn.Linear(in_dim, hidden)
        self.st_attn = STGAttentionBlock(hidden)
        self.temp_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x_seq):
        B, T, N, C = x_seq.shape
        x_emb = self.embed(x_seq)
        # 时序卷积
        x_temp = self.temp_conv(x_emb.permute(0,2,3,1).reshape(B*N, -1, T)).reshape(B,N,-1,T).permute(0,3,1,2)
        out = []
        for t in range(T):
            x_t = self.st_attn(x_temp[:,t], self.adj)
            out.append(x_t)
        out = torch.stack(out, dim=1)
        return self.head(out).permute(0,3,1,2)
