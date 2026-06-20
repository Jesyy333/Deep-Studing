import torch
import torch.nn as nn
from models.dcrnn import DCRNN_GRU, DiffusionConv

class TransferDCRNN(nn.Module):
    """
    跨路网迁移学习DCRNN：预训练LA路网权重，微调湾区PeMS-BAY数据
    冻结底层扩散卷积，仅更新解码器与时序预测头
    """
    def __init__(self, adj_target, source_hidden=64, target_hidden=32, in_dim=3, pred_len=12):
        super().__init__()
        self.N = adj_target.shape[0]
        self.adj = adj_target
        # 源域预训练编码器（冻结权重）
        self.source_enc = DCRNN_GRU(source_hidden, adj_target)
        # 目标域轻量化解码器
        self.target_dec = DCRNN_GRU(target_hidden, adj_target)
        # 维度适配层
        self.dim_adapt = nn.Linear(source_hidden, target_hidden)
        self.proj_in = nn.Linear(in_dim, source_hidden)
        self.proj_out = nn.Linear(target_hidden, 1)
        self.pred_len = pred_len

    def load_pretrained_source(self, ckpt_path):
        """加载LA路网预训练权重，冻结编码器层"""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 只加载编码器相关权重
        source_weight = {}
        for k,v in ckpt.items():
            if "enc_gru" in k:
                new_k = k.replace("enc_gru.","")
                source_weight[new_k] = v
        self.source_enc.load_state_dict(source_weight, strict=False)
        # 冻结源域编码器
        for param in self.source_enc.parameters():
            param.requires_grad = False

    def forward(self, x_seq):
        B, T_in, N, C = x_seq.shape
        h_source = torch.zeros(B, N, self.source_enc.hidden_dim).to(x_seq.device)
        # 冻结的源域编码
        for t in range(T_in):
            x_t = self.proj_in(x_seq[:, t])
            h_source = self.source_enc(x_t, h_source)
        # 迁移维度适配
        h_target = self.dim_adapt(h_source)
        # 目标域解码器预测
        pred_out = []
        dec_in = torch.zeros_like(x_seq[:, -1, :, :1])
        for _ in range(self.pred_len):
            dec_x = self.proj_in(torch.cat([dec_in, torch.zeros_like(x_seq[:,0,:,1:])], -1))
            h_target = self.target_dec(dec_x, h_target)
            y = self.proj_out(h_target)
            pred_out.append(y)
            dec_in = y
        return torch.stack(pred_out, dim=1)
