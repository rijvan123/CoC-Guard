"""Three-stage controlled training (Section III-J, Table A6).

Stage 1  temporal and graph branches are initialised with their auxiliary objectives.
Stage 2  Reliability-Controlled Fusion and both heads are optimised with Eq. (29).
Stage 3  (outside this module) temperatures, prototypes, open-world coefficients,
         thresholds and replay parameters are selected on the frozen development
         partition only.

Optimisation: AdamW (lr 3e-4, weight decay 1e-5), gradient clipping 1.0, five
warm-up epochs followed by cosine decay to 1e-6, at most 120 epochs, early
stopping with patience 15 on the development objective (the checkpoint
criterion uses development data only). Final-test data never reach this module.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Sequence

import numpy as np
import torch

from ..data.collate import iterate_batches
from ..utils.logging import get_logger
from .losses import unified_loss

log = get_logger("trainer")


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_at(epoch: int, base: float, warmup: int, total: int, min_lr: float) -> float:
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))


class Trainer:
    def __init__(self, model, train_cfg: dict, device: torch.device, seed: int = 11, verbose: bool = True):
        self.model = model.to(device)
        self.cfg = dict(train_cfg)
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _epoch(self, samples, optimizer, stage: int, rng: np.random.Generator) -> float:
        self.model.train()
        losses = []
        for batch in iterate_batches(samples, int(self.cfg.get("batch_size", 64)), shuffle=True, rng=rng):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss, _ = unified_loss(self.model, out, batch, self.cfg.get("loss_weights", {}), stage,
                                   float(self.cfg.get("agreement_min_reliability", 0.75)))
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.get("grad_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def dev_objective(self, samples, stage: int = 2) -> float:
        """Development objective used for checkpoint selection (no regulariser)."""
        self.model.eval()
        weights = {k: v for k, v in self.cfg.get("loss_weights", {}).items() if k != "regularization"}
        total, n = 0.0, 0
        for batch in iterate_batches(samples, int(self.cfg.get("batch_size", 64)) * 2, shuffle=False):
            batch = batch.to(self.device)
            out = self.model(batch)
            loss, _ = unified_loss(self.model, out, batch, weights, stage,
                                   float(self.cfg.get("agreement_min_reliability", 0.75)))
            total += float(loss) * batch.batch_size
            n += batch.batch_size
        return total / max(1, n)

    def fit(self, train: Sequence[dict], dev: Sequence[dict]) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        base_lr = float(cfg.get("lr", 3e-4))
        opt = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=float(cfg.get("weight_decay", 1e-5)))
        max_epochs = int(cfg.get("max_epochs", 120))
        stage1 = min(int(cfg.get("stage1_epochs", 10)), max(0, max_epochs - 1))
        warmup = int(cfg.get("warmup_epochs", 5))
        min_lr = float(cfg.get("min_lr", 1e-6))
        patience = int(cfg.get("patience", 15))
        best, best_state, wait = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(max_epochs):
            stage = 1 if epoch < stage1 else 2
            lr = lr_at(epoch, base_lr, warmup, max_epochs, min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            tr = self._epoch(train, opt, stage, rng)
            dv = self.dev_objective(dev, stage)
            self.history.append({"epoch": epoch, "stage": stage, "lr": lr, "train_loss": tr, "dev_objective": dv,
                                 "elapsed_s": time.time() - t0})
            if self.verbose:
                log.info(f"epoch {epoch:3d} stage {stage} lr {lr:.2e} train {tr:.4f} dev {dv:.4f} "
                         f"({time.time() - t0:.0f} s)")
            if stage == 2:
                if dv < best - 1e-6:
                    best, best_state, wait = dv, copy.deepcopy(self.model.state_dict()), 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_dev_objective": best, "epochs": len(self.history), "history": self.history,
                "train_seconds": time.time() - t0}


@torch.no_grad()
def predict(model, samples: Sequence[dict], device: torch.device, batch_size: int = 128,
            keep_events: bool = False) -> dict:
    """Frozen-model inference over a list of samples (order preserved)."""
    model.eval()
    keys = ("logits_m", "probs_m", "z_m", "logits_c", "probs_c", "z_c", "d_tg", "u_T", "u_G", "rho_T", "rho_G",
            "rho_bar", "a_T", "a_G", "g_m", "g_c")
    acc: dict[str, list] = {k: [] for k in keys}
    events: dict[str, list] = {"loc_logits": [], "rho": [], "pool_T": [], "attention_received": []}
    for batch in iterate_batches(samples, batch_size, shuffle=False, bucket=False):
        out = model(batch.to(device))
        for k in keys:
            if k in out and out[k] is not None:
                acc[k].append(out[k].detach().float().cpu().numpy())
        lengths = batch.ev_mask.sum(1).cpu().numpy()
        for k in events:
            if k in out and out[k] is not None and (keep_events or k == "loc_logits"):
                arr = out[k].detach().float().cpu().numpy()
                events[k].extend([arr[i, :int(lengths[i])] for i in range(len(lengths))])
    res = {k: np.concatenate(v) for k, v in acc.items() if v}
    res.update({k: v for k, v in events.items() if v})
    return res
