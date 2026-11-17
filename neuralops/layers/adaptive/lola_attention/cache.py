import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LowRankCache(nn.Module):
    def __init__(self, max_len: int = 128, low_rank_k: int = 16) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.low_rank_k = int(low_rank_k)
        self.register_buffer("_init_flag", torch.zeros(1, dtype=torch.bool), persistent=False)
        self._B_H: int = 0
        self._D: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

        self._recent_k: torch.Tensor | None = None
        self._recent_v: torch.Tensor | None = None
        self._recent_pos: int = 0
        self._recent_filled: int = 0

        self._sparse_k: torch.Tensor | None = None
        self._sparse_v: torch.Tensor | None = None
        self._sparse_pos: int = 0
        self._sparse_filled: int = 0

        self._low_rank_k_buf: torch.Tensor | None = None
        self._low_rank_v_buf: torch.Tensor | None = None

        self._cache_proj: torch.Tensor | None = None
        self._update_step: int = 0

    def _ensure_buffers(self, B_H: int, D: int, device: torch.device, dtype: torch.dtype) -> None:
        if (self._B_H == B_H and self._D == D and self._device == device and self._dtype == dtype and self._recent_k is not None): return
        self._B_H, self._D, self._device, self._dtype = B_H, D, device, dtype
        max_recent, max_sparse = self.max_len, self.max_len // 2
        self._recent_k = torch.zeros(B_H, max_recent, D, device=device, dtype=dtype)
        self._recent_v = torch.zeros(B_H, max_recent, D, device=device, dtype=dtype)
        self._recent_pos, self._recent_filled = 0, 0
        self._sparse_k = torch.zeros(B_H, max_sparse, D, device=device, dtype=dtype)
        self._sparse_v = torch.zeros(B_H, max_sparse, D, device=device, dtype=dtype)
        self._sparse_pos, self._sparse_filled = 0, 0
        self._low_rank_k_buf, self._low_rank_v_buf, self._update_step = None, None, 0

    def reset(self) -> None:
        if self._recent_k is not None:
            self._recent_k.zero_(); self._recent_v.zero_()
            self._sparse_k.zero_(); self._sparse_v.zero_()
        self._recent_pos, self._recent_filled = 0, 0
        self._sparse_pos, self._sparse_filled = 0, 0
        self._low_rank_k_buf, self._low_rank_v_buf, self._update_step = None, None, 0

    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        B, H, N, D = K.shape
        k_flat, v_flat = K.detach().reshape(B * H, N, D), V.detach().reshape(B * H, N, D)
        self._ensure_buffers(B * H, D, k_flat.device, k_flat.dtype)
        max_recent, max_sparse = self._recent_k.size(1), self._sparse_k.size(1)
        take = min(N, max_recent)
        src, src_v = k_flat[:, -take:, :], v_flat[:, -take:, :]
        end = self._recent_pos + take
        if end <= max_recent:
            self._recent_k[:, self._recent_pos : end, :].copy_(src)
            self._recent_v[:, self._recent_pos : end, :].copy_(src_v)
        else:
            first = max_recent - self._recent_pos
            self._recent_k[:, self._recent_pos :, :].copy_(src[:, :first, :])
            self._recent_v[:, self._recent_pos :, :].copy_(src_v[:, :first, :])
            rest = take - first
            self._recent_k[:, :rest, :].copy_(src[:, first:, :])
            self._recent_v[:, :rest, :].copy_(src_v[:, first:, :])
        self._recent_pos = end % max_recent
        self._recent_filled = min(max_recent, self._recent_filled + take)
        sparse_idx = torch.arange(0, N, 4, device=K.device)
        sparse_k, sparse_v = k_flat[:, sparse_idx, :], v_flat[:, sparse_idx, :]
        take_s = min(sparse_k.size(1), max_sparse)
        if take_s > 0:
            src_s, src_sv = sparse_k[:, -take_s:, :], sparse_v[:, -take_s:, :]
            end_s = self._sparse_pos + take_s
            if end_s <= max_sparse:
                self._sparse_k[:, self._sparse_pos : end_s, :].copy_(src_s)
                self._sparse_v[:, self._sparse_pos : end_s, :].copy_(src_sv)
            else:
                first = max_sparse - self._sparse_pos
                self._sparse_k[:, self._sparse_pos :, :].copy_(src_s[:, :first, :])
                self._sparse_v[:, self._sparse_pos :, :].copy_(src_sv[:, :first, :])
                rest = take_s - first
                self._sparse_k[:, :rest, :].copy_(src_s[:, first:, :])
                self._sparse_v[:, :rest, :].copy_(src_sv[:, first:, :])
            self._sparse_pos = end_s % max_sparse
            self._sparse_filled = min(max_sparse, self._sparse_filled + take_s)
        self._update_step += 1
        if self._recent_filled >= self.low_rank_k and (self._update_step % 50 == 1 or self._low_rank_k_buf is None): self._refresh_low_rank()

    def _refresh_low_rank(self) -> None:
        if self._recent_k is None or self._recent_filled < self.low_rank_k: return
        with torch.no_grad():
            valid = self._recent_filled
            if self._recent_pos == 0: rk, rv = self._recent_k[:, :valid, :], self._recent_v[:, :valid, :]
            else:
                if valid <= self._recent_k.size(1):
                    rk = torch.cat([self._recent_k[:, self._recent_pos :, :], self._recent_k[:, : self._recent_pos, :]], dim=1)
                    rv = torch.cat([self._recent_v[:, self._recent_pos :, :], self._recent_v[:, : self._recent_pos, :]], dim=1)
                else: rk, rv = self._recent_k, self._recent_v
            rk_f, rv_f = rk.float(), rv.float()
            D, r = self._D, self.low_rank_k
            if self._cache_proj is None or self._cache_proj.size(0) != D or self._cache_proj.size(1) != r or self._cache_proj.device != rk.device:
                self._cache_proj = torch.empty(D, r, device=rk.device, dtype=torch.float32)
                nn.init.orthogonal_(self._cache_proj)
            try:
                y_k, y_v = torch.matmul(rk_f, self._cache_proj), torch.matmul(rv_f, self._cache_proj)
                q_k, _ = torch.linalg.qr(y_k)
                q_v, _ = torch.linalg.qr(y_v)
                self._low_rank_k_buf = torch.matmul(q_k, torch.matmul(q_k.transpose(-2, -1).contiguous(), rk_f)).to(rk.dtype)
                self._low_rank_v_buf = torch.matmul(q_v, torch.matmul(q_v.transpose(-2, -1).contiguous(), rv_f)).to(rv.dtype)
            except Exception as e: pass

    def get_cached(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._recent_k is None or self._recent_filled == 0: return None, None
        valid = self._recent_filled
        if self._recent_pos == 0: rk, rv = self._recent_k[:, :valid, :], self._recent_v[:, :valid, :]
        else:
            if valid <= self._recent_k.size(1):
                rk = torch.cat([self._recent_k[:, self._recent_pos :, :], self._recent_k[:, : self._recent_pos, :]], dim=1)
                rv = torch.cat([self._recent_v[:, self._recent_pos :, :], self._recent_v[:, : self._recent_pos, :]], dim=1)
            else: rk, rv = self._recent_k, self._recent_v
        tensors_k, tensors_v = [rk], [rv]
        if self._sparse_filled > 0 and self._sparse_k is not None:
            sv = self._sparse_filled
            if self._sparse_pos == 0: sk, svv = self._sparse_k[:, :sv, :], self._sparse_v[:, :sv, :]
            else:
                if sv <= self._sparse_k.size(1):
                    sk = torch.cat([self._sparse_k[:, self._sparse_pos :, :], self._sparse_k[:, : self._sparse_pos, :]], dim=1)
                    svv = torch.cat([self._sparse_v[:, self._sparse_pos :, :], self._sparse_v[:, : self._sparse_pos, :]], dim=1)
                else: sk, svv = self._sparse_k, self._sparse_v
            tensors_k.append(sk); tensors_v.append(svv)
        if self._low_rank_k_buf is not None: tensors_k.append(self._low_rank_k_buf); tensors_v.append(self._low_rank_v_buf)
        return torch.cat(tensors_k, dim=1), torch.cat(tensors_v, dim=1)
