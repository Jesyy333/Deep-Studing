import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # 图卷积：X * W * A
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        return x + self.bias

class TemporalConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size,1), padding=(1,0))

    def forward(self, x):
        # x: [B, C, T, N]
        return self.conv(x)

class STBlock(nn.Module):
    def __init__(self, hidden_dim, adj):
        super().__init__()
        self.adj = adj
        self.temp1 = TemporalConv(hidden_dim, hidden_dim)
        self.graph_conv = GraphConv(hidden_dim, hidden_dim)
        self.temp2 = TemporalConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, C, T, N = x.shape
        residual = x
        # 时序卷积
        x = self.temp1(x)
        # 图卷积交换维度 [B,T,N,C]
        x = x.permute(0,2,3,1)
        x = self.graph_conv(x, self.adj)
        x = x.permute(0,3,1,2)
        x = self.temp2(x)
        return self.norm(x + residual)

class STGCN(nn.Module):
    def __init__(self, adj, in_dim=3, hidden_dim=64, pred_len=12, num_blocks=2):
        super().__init__()
        self.N = adj.shape[0]
        self.adj = adj
        self.proj_in = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1,1))
        self.blocks = nn.ModuleList([STBlock(hidden_dim, adj) for _ in range(num_blocks)])
        self.proj_out = nn.Conv2d(hidden_dim, pred_len, kernel_size=(1,1))

    def forward(self, x_seq):
        # x_seq: [B, T_in, N, C]
        B, T, N, C = x_seq.shape
        x = x_seq.permute(0,3,1,2) # [B,C,T,N]
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        out = self.proj_out(x) # [B, pred_len, T, N]
        return out.permute(0,2,1,3) # [B, T, pred_len, N]
