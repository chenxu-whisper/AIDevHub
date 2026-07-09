import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. 位置编码 Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: [d_model / 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # 偶数维使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer，不参与训练，但会随模型保存
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =========================
# 2. 多头注意力 Multi-Head Attention
# =========================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q/K/V 线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: [B, T_q, d_model]
        key:   [B, T_k, d_model]
        value: [B, T_v, d_model]
        mask:  可 broadcast 到 [B, num_heads, T_q, T_k]
        """

        B = query.size(0)

        # 1. 生成 Q/K/V
        # [B, T, d_model]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. 拆成多头
        # [B, T, d_model] -> [B, T, num_heads, d_head] -> [B, num_heads, T, d_head]
        Q = Q.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 3. 计算注意力分数
        # scores: [B, num_heads, T_q, T_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4. 加 mask
        if mask is not None:
            # mask 中为 0 的位置会被屏蔽
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. softmax 得到注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6. 加权求和 V
        # context: [B, num_heads, T_q, d_head]
        context = torch.matmul(attn, V)

        # 7. 多头拼接
        # [B, num_heads, T_q, d_head] -> [B, T_q, num_heads, d_head]
        context = context.transpose(1, 2).contiguous()

        # [B, T_q, d_model]
        context = context.view(B, -1, self.d_model)

        # 8. 输出投影
        output = self.out_proj(context)

        return output


# =========================
# 3. 前馈神经网络 Feed Forward
# =========================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        return self.net(x)


# =========================
# 4. Encoder Layer
# =========================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        x: [B, src_len, d_model]
        src_mask: [B, 1, 1, src_len]
        """

        # 1. Encoder Self-Attention
        attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=src_mask
        )

        # 2. Add & Norm
        x = self.norm1(x + self.dropout(attn_out))

        # 3. Feed Forward
        ff_out = self.feed_forward(x)

        # 4. Add & Norm
        x = self.norm2(x + self.dropout(ff_out))

        return x


# =========================
# 5. Encoder
# =========================

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        """
        src: [B, src_len]
        """

        # 1. Token Embedding
        x = self.embedding(src) * math.sqrt(self.d_model)

        # 2. Positional Encoding
        x = self.pos_encoding(x)

        # 3. 多层 Encoder
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# =========================
# 6. Decoder Layer
# =========================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        """
        tgt: [B, tgt_len, d_model]
        memory: [B, src_len, d_model]
        tgt_mask: [B, 1, tgt_len, tgt_len]
        src_mask: [B, 1, 1, src_len]
        """

        # 1. Masked Self-Attention
        self_attn_out = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            mask=tgt_mask
        )

        # 2. Add & Norm
        tgt = self.norm1(tgt + self.dropout(self_attn_out))

        # 3. Cross-Attention
        # Q 来自 Decoder，K/V 来自 Encoder
        cross_attn_out = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            mask=src_mask
        )

        # 4. Add & Norm
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))

        # 5. Feed Forward
        ff_out = self.feed_forward(tgt)

        # 6. Add & Norm
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt


# =========================
# 7. Decoder
# =========================

class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        """
        tgt: [B, tgt_len]
        memory: [B, src_len, d_model]
        """

        # 1. Output Embedding
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        # 2. Positional Encoding
        x = self.pos_encoding(x)

        # 3. 多层 Decoder
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)

        return x


# =========================
# 8. 完整 Transformer
# =========================

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        # Linear：映射到目标词表大小
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        """
        src: [B, src_len]
        返回: [B, 1, 1, src_len]
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        tgt: [B, tgt_len]
        返回: [B, 1, tgt_len, tgt_len]
        """

        B, tgt_len = tgt.shape

        # Padding Mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # [B, 1, 1, tgt_len]

        # Causal Mask
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()
        # [tgt_len, tgt_len]

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        # [1, 1, tgt_len, tgt_len]

        # 同时满足 padding mask 和 causal mask
        tgt_mask = tgt_pad_mask & causal_mask
        # [B, 1, tgt_len, tgt_len]

        return tgt_mask

    def forward(self, src, tgt):
        """
        src: [B, src_len]
        tgt: [B, tgt_len]

        返回:
        logits: [B, tgt_len, tgt_vocab_size]
        """

        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # 1. Encoder
        memory = self.encoder(src, src_mask)

        # 2. Decoder
        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

        # 3. Linear 输出 logits
        logits = self.generator(decoder_output)

        return logits


# =========================
# 9. 测试代码
# =========================

if __name__ == "__main__":
    # 假设：
    # 源语言词表大小 10000
    # 目标语言词表大小 12000
    src_vocab_size = 10000
    tgt_vocab_size = 12000
    pad_idx = 0

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
        pad_idx=pad_idx
    )

    # batch size = 2
    # 源序列长度 = 10
    # 目标序列长度 = 8
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))

    logits = model(src, tgt)

    print("logits shape:", logits.shape)
    # 期望输出: [2, 8, 12000] # 2 个样本，每个样本 8 个目标词，每个目标词 12000 个可能的输出
