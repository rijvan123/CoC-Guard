"""MLP baseline (256 hidden units, 3 layers, dropout 0.30, lr 1e-3, batch 128; Table A4)
and a generic trainer for array-based networks (also used by the static CoC-Guard
configuration, the confidence-threshold MLP and OpenMax-MLP)."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, layers: int = 3, dropout: float = 0.30):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x, m=None) -> dict:
        z = self.body(x)
        return {"z_m": z, "logits_m": self.out(z)}


class ArrayTrainer:
    """Mini-batch trainer for ``model(x, m) -> {'logits_m', 'z_m'}`` on (possibly memory-mapped) arrays."""

    def __init__(self, model: nn.Module, device, lr: float = 1e-3, weight_decay: float = 0.0, batch_size: int = 128,
                 max_epochs: int = 100, patience: int = 10, optimizer: str = "adam", warmup_epochs: int = 0,
                 min_lr: float = 1e-6, grad_clip: float | None = None, calibration_weight: float = 0.0,
                 seed: int = 11, scaler=None, verbose: bool = False):
        self.model = model.to(device)
        self.device = device
        self.lr, self.wd, self.bs = lr, weight_decay, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.optimizer = optimizer
        self.warmup, self.min_lr, self.clip = warmup_epochs, min_lr, grad_clip
        self.cal_w = calibration_weight
        self.seed = seed
        self.scaler = scaler
        self.verbose = verbose
        self.history: list = []

    def _xy(self, X, idx):
        x = np.asarray(X[np.sort(idx)] if not isinstance(X, np.ndarray) else X[idx], dtype=np.float32)
        if self.scaler is not None:
            x, m = self.scaler.transform(x)
        else:
            m = np.isfinite(x).astype(np.float32)
            x = np.nan_to_num(x)
        return torch.from_numpy(np.asarray(x, np.float32)).to(self.device), torch.from_numpy(np.asarray(m, np.float32)).to(self.device)

    def _lr(self, epoch: int) -> float:
        if epoch < self.warmup:
            return self.lr * (epoch + 1) / max(1, self.warmup)
        if self.warmup == 0 and self.optimizer == "adam":
            return self.lr
        p = (epoch - self.warmup) / max(1, self.max_epochs - self.warmup)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * min(1.0, p)))

    def fit(self, X, y, train_idx, dev_idx):
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        opt_cls = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        opt = opt_cls(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, wait = float("inf"), None, 0
        t_start = time.time()
        for epoch in range(self.max_epochs):
            for g in opt.param_groups:
                g["lr"] = self._lr(epoch)
            self.model.train()
            perm = rng.permutation(train_idx)
            loss_sum, n_seen = 0.0, 0
            for k in range(0, len(perm), self.bs):
                idx = np.sort(perm[k:k + self.bs])
                x, m = self._xy(X, idx)
                yt = torch.from_numpy(y[idx]).long().to(self.device)
                out = self.model(x, m)
                loss = F.cross_entropy(out["logits_m"], yt)
                if self.cal_w:
                    p = torch.softmax(out["logits_m"], -1)
                    conf, pred = p.max(-1)
                    loss = loss + self.cal_w * (conf.mean() - (pred == yt).float().mean()).abs()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                opt.step()
                loss_sum += float(loss.detach()) * len(idx)
                n_seen += len(idx)
            dev_loss = self.evaluate_loss(X, y, dev_idx)
            self.history.append({"epoch": epoch, "stage": "supervised", "lr": self._lr(epoch),
                                 "train_loss": loss_sum / max(1, n_seen), "dev_objective": dev_loss,
                                 "elapsed_s": time.time() - t_start})
            if dev_loss < best - 1e-6:
                best, best_state, wait = dev_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def fine_tune(self, X, y, idx, epochs: int, lr: float) -> "ArrayTrainer":
        """Chronological adaptation: a few epochs on admitted observations (no early stopping)."""
        y = np.asarray(y)
        idx = np.asarray(idx)
        if len(idx) == 0 or epochs <= 0:
            return self
        rng = np.random.default_rng(self.seed + 7)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.wd)
        for _ in range(epochs):
            self.model.train()
            perm = rng.permutation(idx)
            for k in range(0, len(perm), self.bs):
                sel = np.sort(perm[k:k + self.bs])
                x, m = self._xy(X, sel)
                out = self.model(x, m)
                loss = F.cross_entropy(out["logits_m"], torch.from_numpy(y[sel]).long().to(self.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                opt.step()
        return self

    @torch.no_grad()
    def evaluate_loss(self, X, y, idx) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for k in range(0, len(idx), 4096):
            sel = np.sort(idx[k:k + 4096])
            x, m = self._xy(X, sel)
            out = self.model(x, m)
            tot += float(F.cross_entropy(out["logits_m"], torch.from_numpy(np.asarray(y)[sel]).long().to(self.device),
                                         reduction="sum"))
            n += len(sel)
        return tot / max(1, n)

    @torch.no_grad()
    def predict(self, X, idx) -> dict:
        """Logits and representations in the order of ``idx``."""
        self.model.eval()
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        logits, zs, rho = [], [], []
        for k in range(0, len(idx), 4096):
            sel = idx[order[k:k + 4096]]
            x, m = self._xy(X, sel)
            out = self.model(x, m)
            logits.append(out["logits_m"].cpu().numpy())
            zs.append(out["z_m"].cpu().numpy())
            rho.append(m.mean(dim=1).cpu().numpy())
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        L, Z, R = np.concatenate(logits), np.concatenate(zs), np.concatenate(rho)
        return {"logits_m": L[inv], "z_m": Z[inv], "rho_bar": R[inv]}
