import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, k_diffusion=2):
        super().__init__()
        self.k = k_diffusion
        self.linear = nn.Linear(in_dim * (k_diffusion + 1), out_dim)

    def forward(self, x, adj):
        # 扩散卷积：正向/反向拥堵传播，匹配加州路网拥堵波扩散
        x0 = x
        x1 = torch.matmul(adj, x0)
        x2 = torch.matmul(adj, x1)
        x_cat = torch.cat([x0, x1, x2], dim=-1)
        return self.linear(x_cat)

class DCRNN_GRU(nn.Module):
    def __init__(self, hidden_dim, adj):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adj = adj
        self.conv_r = DiffusionConv(hidden_dim, hidden_dim)
        self.conv_z = DiffusionConv(hidden_dim, hidden_dim)
        self.conv_h = DiffusionConv(hidden_dim, hidden_dim)

    def forward(self, x, h_prev):
        r = torch.sigmoid(self.conv_r(torch.cat([x, h_prev], -1), self.adj))
        z = torch.sigmoid(self.conv_z(torch.cat([x, h_prev], -1), self.adj))
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h_prev], -1), self.adj))
        h = (1 - z) * h_prev + z * h_tilde
        return h

class DCRNN(nn.Module):
    def __init__(self, adj, in_dim=3, hidden_dim=64, pred_len=12):
        super().__init__()
        self.N = adj.shape[0]
        self.adj = adj
        self.enc_gru = DCRNN_GRU(hidden_dim, adj)
        self.dec_gru = DCRNN_GRU(hidden_dim, adj)
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, 1)
        self.pred_len = pred_len

    def forward(self, x_seq):
        B, T_in, N, C = x_seq.shape
        h = torch.zeros(B, N, self.enc_gru.hidden_dim).to(x_seq.device)
        # 编码器
        for t in range(T_in):
            x_t = self.proj_in(x_seq[:, t])
            h = self.enc_gru(x_t, h)
        # 解码器多步预测
        pred_out = []
        dec_in = torch.zeros_like(x_seq[:, -1, :, :1])
        for _ in range(self.pred_len):
            dec_x = self.proj_in(torch.cat([dec_in, torch.zeros_like(x_seq[:,0,:,1:])], -1))
            h = self.dec_gru(dec_x, h)
            y = self.proj_out(h)
            pred_out.append(y)
            dec_in = y
        return torch.stack(pred_out, dim=1)
