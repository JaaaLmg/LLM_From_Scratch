"""
GPT2的模型定义
这个文件严格按照官方的GPT2模型结构进行复现，可以直接使用huggingface中的GPT2参数加载权重
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 先定义模型的主要参数
@dataclass
class GPTConfig:
    block_size: int = 1024   # 原始论文中的上下文长度为1024
    vocab_size: int = 50257  # 原始论文使用的字典大小为50257
    n_layer: int = 12        # 层数为12
    n_head: int = 12         # 注意力头数为12
    n_embd: int = 768        # embedding维度为768
    dropout: float = 0.1     # dropout率
    flash_attn: bool = False     # 是否使用Flash Attention  flash attention在pytorch 1.9.0中才支持



# 定义transformer子块
# GPT中的子块结构与原始transformer略有不同，层归一化在每一层之前进行
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)    # 自注意力模块（多头）
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)    # FFN模块

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
# 定义FFN模块
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)    # 将输入映射到4倍维度的向量空间
        self.gelu = nn.GELU()    # 激活函数
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)  # 将4倍维度的向量空间映射回输入的向量空间
        self.dropout = nn.Dropout(config.dropout)  # dropout层

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)  # 在投影后应用dropout
        return x

# 定义自注意力模块
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0    # 词嵌入维度必须可以被注意力头数整除

        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)     # 从词向量中映射到Q、K、V向量  (B, T, C) -> (B, T, 3C)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # 将concat之后的Q、K、V向量映射回词向量  (B, T, C) -> (B, T, C)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.flash = config.flash_attn

        # attention mask 通过 register_buffer 注册在模型参数中，因为不用计算梯度，节约显存
        self.register_buffer(
            "attention_mask",
            # 创建一个下三角矩阵
            torch.tril(torch.ones(config.block_size,config.block_size))
        )

    def forward(self, x):
        B, T, C = x.size()      # 批量大小，序列长度，词向量维度
        qkv = self.c_attn(x)    # 先求出一整个大的矩阵，再进行分块
        q, k, v = qkv.split(self.n_embd, dim=2)    # q,k,v矩阵的形状均为(B, T, C)， 把最后一维进行划分，即可得到多个头

        # 因为是多头注意力，我们要把每一个头的维度规范为(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)    # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))    # 点积注意力
            att = att.masked_fill(self.attention_mask[:T, :T] == 0, float('-inf'))    # 注意力掩码
            att = F.softmax(att, dim=-1)    # 过softmax得到注意力权重 (B, nh, T, T)
            att = self.attn_dropout(att)
            y = att @ v    # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1,2).contiguous().view(B, T, C)     # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)  contiguous()函数将张量内存连续化
        y = self.c_proj(y)
        return y



# 定义GPT模型
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # 词嵌入
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 位置嵌入
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )    # 一组多层transformer块
        self.ln_f = nn.LayerNorm(config.n_embd)  # 最后的一侧层归一化
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头，将词向量投影回字典，该层没有bias

        # 共享权重
        self.wte.weight = self.lm_head.weight

        # 模型参数初始化
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, model_type):
        """从huggingface加载预训练的GPT-2模型权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("=================================================")
        print("开始加载gpt的权重: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attention_mask')] # 忽略掩码的参数，它不参与梯度的计算

        # 查看模型的参数
        # 查看模型的参数
        # print("=================================================")
        # print("该文件复现的GPT2模型参数如下：")
        # for k,v in sd.items():
        #     print(k, v.shape)

        # 从huggingface/transformers中初始化一个模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 确保我们自己的模型的参数名称和形状与官方模型对齐
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for i,k in enumerate(sd_keys_hf):
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[sd_keys[i]].shape
                with torch.no_grad():
                    sd[sd_keys[i]].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[sd_keys[i]].shape
                with torch.no_grad():
                    sd[sd_keys[i]].copy_(sd_hf[k])

        return model

    def _init_weights(self, module):
        """初始化模型的参数"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 对于LayerNorm层，使用pytorch默认的初始参数（weight=1.0, bias=0.0）
    
    def forward(self, idx, targets=None):
        """
        前向传播
        :param idx: 输入的索引，形状为(B, T)
        :param targets: 是否进行训练，如果是，则传入目标索引
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"无法处理长度为{T}的输入，因为模型只支持长度小于等于{self.config.block_size}的输入"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # 位置嵌入的索引
        pos_emb = self.wpe(pos)    # 获取位置嵌入
        tok_emb = self.wte(idx)    # 获取词嵌入
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


if __name__ == "__main__":
    # 从huggingface加载预训练的GPT-2模型权重
    model = GPT.from_pretrained("gpt2")
    print("=================================================")
    print("与官方模型完美对齐！")
    
    # 生成一段文本
    num_retrun_sequences = 5
    max_length = 50

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")   # 加载gpt2的编码器
    tokens = enc.encode("Now, I'm trying to build a language model on my own,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_retrun_sequences, 1)
    x = tokens.to(device)  # x的形状：(B=5, T)

    # 开始生成
    print("=================================================")
    print("开始生成文本...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]  # 只根据最后一个输出来预测下一词
            probs = F.softmax(logits, dim=-1)   # 获取概率
            topk_probs, topk_indeces = torch.topk(probs, 50, dim=-1)   # 进行topk采样，只在概率最大的50个词中选择
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indeces, -1, ix)
            x = torch.cat((x,xcol), dim=1)   # 将新生成的词追加到输入的末尾

    # 打印生成的文本
    for i in range(0, num_retrun_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)