import torch
import torch.nn as nn
from .config import SOAConfig
from .attention import SOAFusedAttention

class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig, shared_basis):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = shared_basis
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        x_latent = h @ self.basis   # (B,S,r)
        attn_out = self.attn(x_latent, attn_mask)
        x = x + attn_out
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        return x


import torch
import torch.nn as nn
from .config import SOAConfig
# فرض بر این است که SOAFusedAttention وجود دارد، اما ما اینجا ساده‌سازی می‌کنیم

class SOAFusedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k_rank):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank
        
        # وزن‌های تبدیل نهایی (A matrices) که بعداً کپی می‌شوند
        # شکل: [k_rank, head_dim] برای هر سر
        # اما برای سادگی، همه را در یک لایه Linear قرار می‌دهیم
        self.q_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_latent, attn_mask):
        # x_latent: (B, S, k_rank)
        B, S, _ = x_latent.shape
        
        # تبدیل به Q, K, V با استفاده از وزن‌های A (که در مهاجرت پر شده‌اند)
        q = self.q_proj(x_latent) # (B, S, H)
        k = self.k_proj(x_latent) # (B, S, H)
        v = self.v_proj(x_latent) # (B, S, H)
        
        # تقسیم به سرها
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # محاسبه توجه
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores + attn_mask
            
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(output)


class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig, shared_basis):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        
        # پایه مشترک (ثابت)
        self.basis = shared_basis # شکل: [hidden_size, k_rank]
        
        # لایه توجه با پایه مشترک
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask):
        # 1. نرمالایزیشن
        h = self.ln1(x)
        
        # 2. کاهش ابعاد به فضای مشترک (اینجا دقیقاً کاری که تو خواستی)
        # x: (B, S, H) -> basis: (H, R) -> result: (B, S, R)
        # این همان ضرب در بردارهای ویژه است
        x_latent = h @ self.basis 
        
        # 3. توجه در فضای فشرده
        attn_out = self.attn(x_latent, attn_mask)
        
        # 4. جمع residual
        x = x + attn_out
        
        # 5. لایه دوم
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        
        return x


class SOABert(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg

        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # پایه مشترک (Parameter نیست، ثابت است)
        self.basis = nn.Parameter(torch.empty(cfg.hidden_size, cfg.k_rank), requires_grad=False)

        self.layers = nn.ModuleList([SOABlock(cfg, self.basis) for _ in range(cfg.num_hidden_layers)])
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)

        mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4

        for layer in self.layers:
            x = layer(x, mask)

        cls = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls))
        logits = self.cls(pooled)
        return logits

import torch
import torch.nn as nn
from .config import SOAConfig

class SVDBasedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k_rank=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank if k_rank else hidden_size

    def forward(self, x, attn_mask):
        """
        x: [B, S, H]
        
        1. SVD روی ورودی
        2. V = U @ S (بردار ویژه ضربدر مقدار ویژه)
        3. Attention با V
        """
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)  # [N, H]
        
        # 1. SVD
        U, S_vals, Vh = torch.linalg.svd(x_flat, full_matrices=False)
        
        # 2. انتخاب top-k
        U_k = U[:, :self.k_rank]        # [N, k]
        S_k = S_vals[:self.k_rank]      # [k]
        Vh_k = Vh[:self.k_rank, :]      # [k, H]
        
        # 3. ساخت V_approx = U_k @ diag(S_k)
        #    یعنی هر ستون U را در مقدار ویژه متناظرش ضرب کن
        V_approx = U_k * S_k.unsqueeze(0)  # [N, k]
        
        # reshape برای attention
        V = V_approx.reshape(B, S, self.k_rank)  # [B, S, k]
        
        q = V
        k = V
        v = V
        
        # تقسیم به head‌ها
        head_dim = self.k_rank // self.num_heads
        q = q.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, S, self.k_rank)
        
        return output
    
class SVDBasedBlock(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        
        self.attn = SVDBasedAttention(cfg.hidden_size, cfg.num_attention_heads, k_rank=cfg.k_rank)
        
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        
        # ✅ اینجا SVD دینامیک انجام می‌شود
        attn_out = self.attn(h, attn_mask)
        
        # اگر خروجی توجه بعدش کمتر از Hidden_Size بود، باید به Hidden_Size برگردد
        # برای سادگی، فرض می‌کنیم lnn بعدی با k_rank کار می‌کند یا یک لایه Projection اضافه است.
        # اما در اینجا برای جلوگیری از خطا، فرض می‌کنیم k_rank == hidden_size 
        # یا اینکه FFN با ورودی k_rank سازگار است.
        
        x = x + attn_out
        
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        
        return x


class SVDBasedBERT(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg

        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.layers = nn.ModuleList([SVDBasedBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)

        mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4

        for layer in self.layers:
            x = layer(x, mask)

        cls = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls))
        logits = self.cls(pooled)
        return logits