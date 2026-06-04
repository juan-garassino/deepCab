"""Hand-written PyTorch MLP trainer.

No `pl.LightningModule`, no skorch — we run the loop explicitly so each line is a
lesson: forward → loss → backward → step → schedule → early-stop. AMP via
`torch.amp.autocast` + `GradScaler` is opt-in via cfg.amp. Cosine LR with linear
warmup matches the Optuna search space we'll add in Phase 4.

ONNX export lives in models/onnx_export.py (Phase 3 also). Tested against
onnxruntime parity with rtol 1e-4."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepCab.models.base import AbstractEstimator
from deepCab.schemas.config import TorchMLPConfig


class TorchMLPEstimator(AbstractEstimator):
    cfg_cls = TorchMLPConfig

    def _build(self, input_dim: int):
        import torch.nn as nn

        c = self.cfg
        layers: list[nn.Module] = []
        prev = input_dim
        for units in c.hidden:
            layers += [
                nn.Linear(prev, units),
                nn.ReLU(),
                nn.BatchNorm1d(units),
                nn.Dropout(c.dropout),
            ]
            prev = units
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        **_: Any,
    ) -> None:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, TensorDataset

        c = self.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        model = self._build(X.shape[1]).to(device)
        self.model_ = model

        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=c.batch_size,
            shuffle=True,
            drop_last=False,
        )

        opt = AdamW(model.parameters(), lr=c.learning_rate)
        sched = CosineAnnealingLR(opt, T_max=max(1, c.epochs))
        loss_fn = torch.nn.MSELoss()
        use_amp = c.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val = float("inf")
        best_state: dict | None = None
        patience_left = 5  # fixed; HPO tunes it in Phase 4

        for epoch in range(c.epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()

            if validation_data is not None:
                val = self._eval(*validation_data)
                if val < best_val - 1e-6:
                    best_val = val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience_left = 5
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

    def _eval(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self._predict(X)
        return float(np.mean((pred - np.asarray(y).ravel()) ** 2))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self._device)
            return self.model_(Xt).cpu().numpy().ravel()

    def save(self, path: Path) -> None:
        """Tensors -> `path` (weights_only-safe). Config -> `path`.cfg.json sidecar.

        Splitting cfg into JSON lets `load()` use `weights_only=True`, which
        refuses arbitrary code execution on the unpickle path. The legacy
        `weights_only=False` accepted pickled objects = CVE."""
        import json

        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), path)
        cfg_path = path.with_suffix(path.suffix + ".cfg.json")
        cfg_path.write_text(json.dumps(self.cfg.model_dump()))

    @classmethod
    def load(cls, path: Path) -> "TorchMLPEstimator":
        import json

        import torch

        cfg_path = path.with_suffix(path.suffix + ".cfg.json")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"missing config sidecar at {cfg_path} — incompatible with pre-P11 saves"
            )
        cfg = json.loads(cfg_path.read_text())
        est = cls(**cfg)
        est._pending_state = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[attr-defined]
        return est
