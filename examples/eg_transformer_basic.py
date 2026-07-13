import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 这个示例实现了一个“教学版” Transformer。
# 目标不是覆盖工业级训练细节，而是尽量把核心结构讲清楚：
# 1. 位置编码负责给 token 注入位置信息；
# 2. 多头注意力负责建模 token 之间的依赖关系；
# 3. 前馈网络负责对每个位置做非线性变换；
# 4. Encoder 负责编码输入序列；
# 5. Decoder 在看见历史目标 token 的前提下生成下一个位置的表示。


# =========================
# 1. 位置编码 Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    """
    使用固定的正弦/余弦位置编码。

    为什么需要位置编码：
    - Transformer 的自注意力本身不包含“顺序”概念；
    - 如果不给模型额外的位置特征，模型只知道有哪些 token，
      却不知道它们在句子中的先后关系；
    - 因此需要把“位置向量”加到 token embedding 上。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe 的形状先初始化为 [max_len, d_model]。
        # 其中每一行对应一个位置，每一列对应 embedding 的一个维度。
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        # 例如当 max_len=5 时，值大致是：
        # [[0], [1], [2], [3], [4]]
        # 后面会与 div_term 做广播相乘，得到每个位置、每个维度的角度值。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 原始公式分母为 10000^(2i/d_model)。
        # 这里改写成 exp(log()) 形式，数值计算更方便：
        # exp(-log(10000) * 2i / d_model)
        # 只对偶数维生成一次，奇数维复用相同频率。
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维使用 sin，奇数维使用 cos。
        # 这样做的好处是：
        # - 不同位置拥有不同的编码模式；
        # - 相邻位置之间有平滑变化；
        # - 模型可以通过线性组合学习到相对位置信息。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展 batch 维，变成 [1, max_len, d_model]。
        # 后续与输入 x 相加时，可以自动 broadcast 到整个 batch。
        pe = pe.unsqueeze(0)

        # register_buffer 表示：
        # - pe 会跟随模型保存/加载；
        # - pe 不会被优化器更新；
        # - pe 会随着模型 .to(device) 自动移动到对应设备。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]

        返回:
            加入位置编码并做 dropout 后的结果，形状仍为 [B, T, d_model]
        """

        # 根据当前序列长度截取对应的位置编码。
        # self.pe[:, :seq_len, :] 的形状为 [1, seq_len, d_model]。
        seq_len = x.size(1)

        # 把 token embedding 和 positional encoding 逐元素相加。
        # 这里采用“相加”而不是拼接，是 Transformer 经典做法。
        x = x + self.pe[:, :seq_len, :]

        # 最后做一次 dropout，降低过拟合风险。
        return self.dropout(x)


# =========================
# 2. 多头注意力 Multi-Head Attention
# =========================

class MultiHeadAttention(nn.Module):
    """
    标准多头注意力模块。

    输入 query / key / value 后，流程如下：
    1. 先分别线性映射到 Q/K/V；
    2. 按 head 拆分；
    3. 每个 head 独立做 scaled dot-product attention；
    4. 拼接多个 head 的结果；
    5. 再做一次线性映射得到最终输出。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # 每个 head 分到的维度必须是整数。
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q / K / V 分别使用独立线性层映射。
        # 输入输出维度都为 d_model，但参数彼此不同。
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 多头结果拼接之后，再用 out_proj 混合各个 head 的信息。
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: [B, T_q, d_model]
            key:   [B, T_k, d_model]
            value: [B, T_v, d_model]
            mask:  可 broadcast 到 [B, num_heads, T_q, T_k]

        返回:
            output: [B, T_q, d_model]
        """

        B = query.size(0)

        # 1. 线性投影，得到 Q / K / V。
        # 此时仍然是“单头”表示，形状都还是 [B, T, d_model]。
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. 拆成多头。
        # view 后得到 [B, T, num_heads, d_head]，
        # transpose 后得到 [B, num_heads, T, d_head]，
        # 方便后续以 head 为维度并行计算注意力。
        Q = Q.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 3. 计算注意力分数。
        # K.transpose(-2, -1) 把最后两个维度从 [T_k, d_head] 变成 [d_head, T_k]。
        # matmul 后得到每个 query 位置对所有 key 位置的相关性分数。
        # 再除以 sqrt(d_head) 是为了避免维度大时点积数值过大，导致 softmax 梯度过小。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4. 应用 mask。
        # mask 为 0 的位置说明“不允许关注”，例如：
        # - Encoder 中用于屏蔽 padding；
        # - Decoder 中用于屏蔽未来时刻。
        # 被屏蔽的位置填为 -inf，softmax 后概率会接近 0。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. 对最后一个维度做 softmax，得到注意力权重。
        # 最后一个维度对应的是“当前 query 对所有 key 的分布”。
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6. 用注意力权重对 V 做加权求和。
        # 输出 context 的形状是 [B, num_heads, T_q, d_head]。
        context = torch.matmul(attn, V)

        # 7. 把多个 head 的结果拼接回来。
        # 先变成 [B, T_q, num_heads, d_head]，
        # 再 reshape 成 [B, T_q, d_model]。
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, -1, self.d_model)

        # 8. 经过输出线性层，融合所有 head 的信息。
        output = self.out_proj(context)
        return output


# =========================
# 3. 前馈神经网络 Feed Forward
# =========================

class FeedForward(nn.Module):
    """
    Transformer 中的位置前馈网络。

    注意这里不是跨时间步的网络，而是对序列中“每个位置”独立地
    做两层线性变换 + 非线性激活。可以理解为：
    - 注意力层负责“不同 token 之间交流信息”；
    - FFN 负责“每个 token 自己再做一次特征变换”。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            # 先把维度从 d_model 扩大到更高维 d_ff。
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 再投影回 d_model，方便与残差连接相加。
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        参数:
            x: [B, T, d_model]
        """
        return self.net(x)


# =========================
# 4. Encoder Layer
# =========================

class EncoderLayer(nn.Module):
    """
    单层 Encoder Block。

    结构顺序：
    1. Self-Attention
    2. Add & Norm
    3. Feed Forward
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        参数:
            x: [B, src_len, d_model]
            src_mask: [B, 1, 1, src_len]
        """

        # 1. Encoder 自注意力。
        # 因为是 self-attention，所以 Q / K / V 都来自同一个 x。
        # 这意味着源序列中每个位置都可以和其他位置交互。
        attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=src_mask
        )

        # 2. 残差连接 + LayerNorm。
        # 残差连接有助于深层网络训练稳定，LayerNorm 有助于数值分布稳定。
        x = self.norm1(x + self.dropout(attn_out))

        # 3. 位置前馈网络。
        ff_out = self.feed_forward(x)

        # 4. 再做一次残差连接 + LayerNorm。
        x = self.norm2(x + self.dropout(ff_out))

        return x


# =========================
# 5. Encoder
# =========================

class Encoder(nn.Module):
    """
    由多层 EncoderLayer 堆叠而成的编码器。

    输入:
        token id 序列 [B, src_len]
    输出:
        编码后的上下文表示 [B, src_len, d_model]
    """

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

        # 把离散 token id 映射成连续向量。
        self.embedding = nn.Embedding(src_vocab_size, d_model)

        # 给 embedding 注入位置信息。
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 堆叠多个 EncoderLayer。
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        """
        参数:
            src: [B, src_len]
        """

        # 1. token embedding。
        # 乘以 sqrt(d_model) 是 Transformer 论文里的常见做法，
        # 用来让 embedding 的数值尺度更合适。
        x = self.embedding(src) * math.sqrt(self.d_model)

        # 2. 叠加位置编码。
        x = self.pos_encoding(x)

        # 3. 依次通过多层 Encoder。
        # 每一层都会进一步融合上下文信息。
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# =========================
# 6. Decoder Layer
# =========================

class DecoderLayer(nn.Module):
    """
    单层 Decoder Block。

    与 EncoderLayer 相比多了一个 Cross-Attention：
    1. Masked Self-Attention
    2. Add & Norm
    3. Cross-Attention
    4. Add & Norm
    5. Feed Forward
    6. Add & Norm
    """

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
        参数:
            tgt: [B, tgt_len, d_model]
            memory: [B, src_len, d_model]
            tgt_mask: [B, 1, tgt_len, tgt_len]
            src_mask: [B, 1, 1, src_len]
        """

        # 1. Decoder 内部的 masked self-attention。
        # tgt_mask 会阻止当前位置看到“未来 token”，从而保证自回归生成成立。
        self_attn_out = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            mask=tgt_mask
        )

        # 2. Add & Norm。
        tgt = self.norm1(tgt + self.dropout(self_attn_out))

        # 3. Cross-Attention。
        # 这里非常关键：
        # - Q 来自 Decoder 当前状态 tgt；
        # - K / V 来自 Encoder 输出 memory。
        # 因此 Decoder 可以在生成目标序列时“对齐并读取”源序列信息。
        cross_attn_out = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            mask=src_mask
        )

        # 4. Add & Norm。
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))

        # 5. Feed Forward。
        ff_out = self.feed_forward(tgt)

        # 6. Add & Norm。
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt


# =========================
# 7. Decoder
# =========================

class Decoder(nn.Module):
    """
    由多层 DecoderLayer 堆叠而成的解码器。

    输入:
        tgt token id 序列 [B, tgt_len]
        encoder memory [B, src_len, d_model]
    输出:
        解码后的隐藏表示 [B, tgt_len, d_model]
    """

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
        参数:
            tgt: [B, tgt_len]
            memory: [B, src_len, d_model]
        """

        # 1. 目标序列 embedding。
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        # 2. 注入位置信息。
        x = self.pos_encoding(x)

        # 3. 依次通过多层 Decoder。
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)

        return x


# =========================
# 8. 完整 Transformer
# =========================

class Transformer(nn.Module):
    """
    一个完整的 Encoder-Decoder Transformer。

    流程概览：
    1. src -> Encoder，得到 memory；
    2. tgt + mask -> Decoder，并结合 memory 做交叉注意力；
    3. 通过线性层映射到目标词表，得到每个位置的 logits。
    """

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

        # 把 Decoder 输出的隐藏状态映射到目标词表大小，
        # 得到每个位置对所有词的打分（logits）。
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # pad_idx 用于构造 padding mask。
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        """
        参数:
            src: [B, src_len]

        返回:
            src_mask: [B, 1, 1, src_len]

        说明:
        - 非 padding 位置为 True；
        - padding 位置为 False；
        - 这个形状可以广播到注意力分数 [B, num_heads, T_q, src_len]。
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        参数:
            tgt: [B, tgt_len]

        返回:
            tgt_mask: [B, 1, tgt_len, tgt_len]

        tgt_mask 需要同时满足两个约束：
        1. 不能看见 padding；
        2. 不能看见未来时刻。
        """

        B, tgt_len = tgt.shape

        # 1. padding mask。
        # 形状: [B, 1, 1, tgt_len]
        # 只保留非 pad 的 key 位置。
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # 2. causal mask（下三角矩阵）。
        # 例如 tgt_len=4 时，大致为：
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        # 表示第 i 个位置只能看到自己和自己之前的位置。
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()

        # 扩成 [1, 1, tgt_len, tgt_len]，便于与 batch 维广播。
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        # 3. 同时应用 padding mask 和 causal mask。
        # 最终得到 [B, 1, tgt_len, tgt_len]。
        tgt_mask = tgt_pad_mask & causal_mask

        return tgt_mask

    def forward(self, src, tgt):
        """
        参数:
            src: [B, src_len]
            tgt: [B, tgt_len]

        返回:
            logits: [B, tgt_len, tgt_vocab_size]
        """

        # 先构造源序列 mask 和目标序列 mask。
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # 1. 编码源序列，得到 memory。
        memory = self.encoder(src, src_mask)

        # 2. 解码目标序列。
        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )

        # 3. 投影到目标词表，输出 logits。
        logits = self.generator(decoder_output)

        return logits


# =========================
# 9. 测试代码
# =========================

if __name__ == "__main__":
    # 下面这段代码只是做一次最小可运行验证，
    # 用随机输入检查模型前向传播是否通畅、输出维度是否符合预期。

    # 假设：
    # - 源语言词表大小为 10000；
    # - 目标语言词表大小为 12000；
    # - 0 号 token 表示 padding。
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

    # 随机构造一个 batch：
    # - batch size = 2
    # - 源序列长度 = 10
    # - 目标序列长度 = 8
    # 这里用 randint(1, vocab_size, ...) 避开 0，
    # 是为了让这个简单例子里默认不会出现 padding。
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))

    # 前向传播，得到每个目标位置在整个目标词表上的预测分数。
    logits = model(src, tgt)

    print("logits shape:", logits.shape)
    # 期望输出: [2, 8, 12000]
    # 含义是：
    # - 2 个样本；
    # - 每个样本有 8 个目标位置；
    # - 每个位置对应 12000 个词表候选分数。