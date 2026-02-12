"""
GPT2模型的训练脚本，包括数据集的加载、模型训练、模型保存等
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken

from model import GPTConfig, GPT

# 手写一个简易的数据加载器
class DataLoaderLite:
    def __init__(self, data_dir, B, T):
        self.B = B   # 批量大小
        self.T = T   # 句子长度
        self.data_dir = data_dir   # 数据集路径

        with open(data_dir, "r") as f:
            text = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"数据集大小：{len(self.tokens)}")
        print(f"一个epoch的数据批量: {len(self.tokens) // (B*T)}")

        self.current_position = 0    # 当前位置

    def next_batch(self):
        """取出一个批量的训练数据"""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = buf[:-1].view(B, T)    # 输入
        y = buf[1:].view(B, T)     # 目标
        self.current_position += B*T
        # 如果最后剩下的一块数据长度不足一个batch，则将其舍弃
        if self.current_position + (B*T+1) >= len(self.tokens):
            self.current_position = 0
        return x, y



if __name__ == "__main__":
    
    # ==============下面的内容用于调试训练流程，基于tinyshakespeare数据集=============
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    B, T = 4, 32
    data_dir = "../data/input.txt"
    print("==============   开始加载数据...  ================")
    dataloader = DataLoaderLite(data_dir, B, T)
    model = GPT(GPTConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    print("================   开始训练...  ==================")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad()
        x, y = dataloader.next_batch()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step: {i}, loss:{loss.item()}")