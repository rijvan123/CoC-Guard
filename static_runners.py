"""Static-track methods (BODMAS, EMBER): LR, MLP, LightGBM and the CoC-Guard static configuration."""
from __future__ import annotations

import numpy as np
import torch

from ..data.preprocessing import StaticScaler
from ..evaluation.metrics import classification_metrics
from ..models.baselines.classic import LightGBMBaseline, LogisticRegressionBaseline
from ..models.baselines.mlp import MLP, ArrayTrainer
from ..models.static import StaticCoCGuard
from ..openworld.calibration import fit_temperature, softmax_np


def fit_scaler(X, train_idx) -> StaticScaler:
    return StaticScaler().fit(X, train_idx)


def _dense(X, idx, scaler) -> np.ndarray:
    out = []
    for k in range(0, len(idx), 20000):
        out.append(scaler.transform(np.asarray(X[np.asarray(idx[k:k + 20000])], dtype=np.float32))[0])
    return np.concatenate(out)


def run_static_method(method: str, X, y, train_idx, dev_idx, test_idx, n_classes: int, cfg, device, seed: int,
                      scaler: StaticScaler, max_train: int | None = None) -> tuple[dict, dict]:
    """Train ``method`` on the training rows, calibrate on development rows, evaluate on test rows."""
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx)
    if max_train and len(tr) > max_train:
        tr = np.sort(rng.choice(tr, size=max_train, replace=False))
    y = np.asarray(y)
    if method in ("lightgbm", "logreg"):
        Xtr, Xdv, Xte = _dense(X, tr, scaler), _dense(X, dev_idx, scaler), _dense(X, test_idx, scaler)
        model = (LightGBMBaseline(seed=seed) if method == "lightgbm" else LogisticRegressionBaseline(seed=seed))
        model.fit(Xtr, y[tr], Xdv, y[dev_idx])
        probs = model.predict_proba(Xte)
        dev_probs = model.predict_proba(Xdv)
        extra = {"z": None, "dev_probs": dev_probs}
    else:
        torch.manual_seed(seed)
        in_dim = X.shape[1]
        if method == "mlp":
            net = MLP(in_dim, n_classes, hidden=256, layers=3, dropout=0.30)
            trainer = ArrayTrainer(net, device, lr=1e-3, batch_size=128, max_epochs=int(cfg.training.get("static_max_epochs", 100)),
                                   patience=10, optimizer="adam", seed=seed, scaler=scaler)
        else:  # CoC-Guard static configuration
            net = StaticCoCGuard(in_dim, n_classes, hidden=tuple(cfg.model.static_hidden), d=int(cfg.model.hidden_dim),
                                 dropout=float(cfg.model.dropout))
            t = cfg.training
            trainer = ArrayTrainer(net, device, lr=float(t.get("static_lr", t.lr)), weight_decay=float(t.weight_decay),
                                   batch_size=int(t.get("static_batch_size", t.batch_size)),
                                   max_epochs=int(t.get("static_max_epochs", 100)), patience=int(t.patience), optimizer="adamw",
                                   warmup_epochs=int(t.warmup_epochs), min_lr=float(t.min_lr), grad_clip=float(t.grad_clip),
                                   calibration_weight=float(t.loss_weights.calibration), seed=seed, scaler=scaler)
        trainer.fit(X, y, tr, np.asarray(dev_idx))
        from .runners import write_training_log
        write_training_log(method, seed, {"history": trainer.history, "epochs": len(trainer.history)}, net,
                           {"n_train": len(tr), "n_dev": len(dev_idx)})
        P_dev = trainer.predict(X, dev_idx)
        T = fit_temperature(P_dev["logits_m"], y[dev_idx]) if method == "cocguard_static" else 1.0
        P = trainer.predict(X, test_idx)
        probs = softmax_np(P["logits_m"], T)
        extra = {"z": P["z_m"], "dev_probs": softmax_np(P_dev["logits_m"], T), "dev_z": P_dev["z_m"], "temperature": T,
                 "trainer": trainer, "rho_bar": P["rho_bar"]}
    yt = y[test_idx]
    pred = probs.argmax(1)
    m = classification_metrics(yt, pred, probs, labels=list(range(n_classes)))
    return m, {"y_true": yt, "y_pred": pred, "probs": probs, **extra}
