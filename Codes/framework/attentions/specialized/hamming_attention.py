import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hamming")
class HammingAttention(BaseAttention):
    """
    Hamming Attention Distillation (HAD)
    Paper: https://arxiv.org/abs/2502.01770
    """

    def __init__(self, config, rank=64, distill_temperature=1.0, sparsity_ratio=0.5):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.distill_temperature = distill_temperature
        self.sparsity_ratio = sparsity_ratio

        # Standard Q, K, V projections (full precision)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Learnable scaling factors for each head/token (same idea as BinaryFormer)
        self.q_scale = nn.Parameter(torch.ones(self.num_heads, 1))
        self.k_scale = nn.Parameter(torch.ones(self.num_heads, 1))

        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # For distillation (teacher attention stored during forward)
        self.teacher_attention = None

    def _binarize(self, x, scale):
        """Binarize a tensor to approximate {-1, +1} with a steep tanh for gradients."""
        x = x * scale.view(1, -1, 1, 1)           # (B, H, N, D) * (H, 1)
        return torch.tanh(x * 5.0)                 # steep tanh approximates sign

    def _hamming_distance(self, q_b, k_b):
        """
        Hamming distance for {-1, +1} vectors.
        Hamming distance = (D - dot_product) / 2
        """
        dot = torch.matmul(q_b, k_b.transpose(-2, -1))   # (B, H, N, N)
        return (self.head_dim - dot) / 2

    def forward(self, hidden_states, attention_mask=None, teacher_attention=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Full-precision Q, K, V
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 2. Binarize Q and K
        Q_b = self._binarize(Q, self.q_scale)
        K_b = self._binarize(K, self.k_scale)

        # 3. Compute Hamming distance attention scores
        #    Score = - Hamming_distance(Q_b, K_b) / sqrt(head_dim)
        scores = -self._hamming_distance(Q_b, K_b) / (self.head_dim ** 0.5)

        # 4. Apply attention mask (if any)
        if attention_mask is not None:
            scores = scores + attention_mask.view(batch_size, 1, 1, seq_len)

        # 5. Sparsification: keep only top-k scores per row (optional)
        if self.sparsity_ratio > 0 and self.training:
            k = int(seq_len * self.sparsity_ratio)
            topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
            mask = torch.zeros_like(scores).scatter(-1, topk_idx, 1.0)
            scores = scores * mask + (1 - mask) * (-1e9)

        # 6. Attention probabilities
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # 7. Context aggregation
        context = torch.matmul(probs, V.permute(0, 2, 1, 3))
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)

        # 8. Distillation loss (if teacher attention provided)
        distill_loss = None
        if teacher_attention is not None and self.training:
            teacher_attn = teacher_attention.detach()
            distill_loss = self._distillation_loss(probs, teacher_attn, self.distill_temperature)

        output = self.out_proj(context)
        return (output, distill_loss) if distill_loss is not None else (output, None)

    def _distillation_loss(self, student_probs, teacher_probs, temperature=1.0):
        """KL divergence between student and teacher attention distributions."""
        student_log = torch.log(student_probs + 1e-8) / temperature
        teacher_log = torch.log(teacher_probs + 1e-8) / temperature
        loss = F.kl_div(
            F.log_softmax(student_log, dim=-1),
            F.softmax(teacher_log, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        return loss