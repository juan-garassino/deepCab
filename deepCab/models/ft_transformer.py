"""Feature Tokenizer + Transformer (Gorishniy et al. 2021).

Minimal from-scratch impl — no rtdl dep — so we control the attention path that
will be exported to ONNX in Phase 7. Architecture:

    [batch, n_features] of floats
        │
        ▼ FeatureTokenizer:  x_i  -> x_i * W_i + b_i ∈ R^d   (per feature)
    [batch, n_features, d_token]
        │
        ▼ prepend learned CLS token
    [batch, n_features+1, d_token]
        │
        ▼ N × { LayerNorm -> MHA + residual -> LayerNorm -> FFN + residual }
        │
        ▼ take CLS slot, linear -> 1 scalar
    [batch]

Hardcoded n_heads=8, ffn_hidden = d_token * 2 — cfg keeps the four levers
(d_token, n_blocks, attention_dropout, ffn_dropout) that matter for HPO. Torch
≥ 2.1 needed for clean SDPA -> ONNX opset 17+ export."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepCab.models.base import AbstractEstimator
from deepCab.schemas.config import FTTransformerConfig

N_HEADS = 8
FFN_MULT = 2


def _build_module(input_dim: int, c: FTTransformerConfig):
    import torch
    import torch.nn as nn

    if c.d_token % N_HEADS != 0:
        raise ValueError(f"d_token={c.d_token} must be divisible by n_heads={N_HEADS}")

    class FeatureTokenizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(input_dim, c.d_token))
            self.bias = nn.Parameter(torch.empty(input_dim, c.d_token))
            self.cls = nn.Parameter(torch.empty(1, 1, c.d_token))
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            nn.init.kaiming_uniform_(self.bias, a=5**0.5)
            nn.init.kaiming_uniform_(self.cls, a=5**0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, F) -> (B, F, d)  via x.unsqueeze(-1) * W + b
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(x.shape[0], -1, -1)
            return torch.cat([cls, tokens], dim=1)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(c.d_token)
            self.attn = nn.MultiheadAttention(
                c.d_token, N_HEADS, dropout=c.attention_dropout, batch_first=True
            )
            self.ln2 = nn.LayerNorm(c.d_token)
            self.ffn = nn.Sequential(
                nn.Linear(c.d_token, c.d_token * FFN_MULT),
                nn.GELU(),
                nn.Dropout(c.ffn_dropout),
                nn.Linear(c.d_token * FFN_MULT, c.d_token),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x

    class FTTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = FeatureTokenizer()
            self.blocks = nn.ModuleList([Block() for _ in range(c.n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(c.d_token), nn.Linear(c.d_token, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.tokenizer(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h[:, 0]).squeeze(-1)

    return FTTransformer()


class FTTransformerEstimator(AbstractEstimator):
    cfg_cls = FTTransformerConfig

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        **_: Any,
    ) -> None:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        c = self.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._input_dim = X.shape[1]

        self.model_ = _build_module(X.shape[1], c).to(device)

        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y).ravel(), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xt, yt), batch_size=c.batch_size, shuffle=True
        )

        opt = AdamW(self.model_.parameters(), lr=c.learning_rate)
        loss_fn = torch.nn.MSELoss()

        for _ in range(c.epochs):
            self.model_.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self._device)
            return self.model_(Xt).cpu().numpy().ravel()

    def save(self, path: Path) -> None:
        """Tensors -> `path` (weights_only-safe). Config + input_dim -> JSON sidecar.
        See models/torch_mlp.py save() for the rationale."""
        import json

        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), path)
        cfg_path = path.with_suffix(path.suffix + ".cfg.json")
        cfg_path.write_text(
            json.dumps({"cfg": self.cfg.model_dump(), "input_dim": self._input_dim})
        )

    @classmethod
    def load(cls, path: Path) -> "FTTransformerEstimator":
        import json

        import torch

        cfg_path = path.with_suffix(path.suffix + ".cfg.json")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"missing config sidecar at {cfg_path} — incompatible with pre-P11 saves"
            )
        blob = json.loads(cfg_path.read_text())
        est = cls(**blob["cfg"])
        est._input_dim = blob["input_dim"]  # type: ignore[attr-defined]
        est._device = torch.device("cpu")  # type: ignore[attr-defined]
        est.model_ = _build_module(est._input_dim, est.cfg)
        est.model_.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return est
