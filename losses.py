"""Unified learning objective (Eq. 29).

    L = lambda_m L_m + lambda_c L_c + lambda_l L_loc + lambda_a L_agree + lambda_cal L_cal + lambda_reg ||Theta||^2

* ``L_m`` / ``L_c``  cross-entropy of the malware / custody heads;
* ``L_loc``         listwise localisation loss over event positions of compromised
                    histories (targets are event-level violation positions, never
                    malware-family labels);
* ``L_agree``       Jensen-Shannon agreement between temporal and graph auxiliary
                    predictions, applied only when both branches are available and
                    reliable (contradictory or incomplete branches are not forced
                    to agree);
* ``L_cal``         difference between confidence and accuracy (DCA), a
                    differentiable calibration penalty;
* ``||Theta||^2``   explicit L2 regularisation.

The branch-level auxiliary cross-entropies are the stage-1 objective used to
initialise the temporal and graph branches and are kept with a small weight in
stage 2 so that uncertainty and disagreement (Eq. 15) remain meaningful.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.cocguard import js_divergence


def dca_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    conf, pred = p.max(dim=-1)
    acc = (pred == y).float()
    return (conf.mean() - acc.mean()).abs()


def localization_loss(loc_logits: torch.Tensor, loc: torch.Tensor, ev_mask: torch.Tensor,
                      loc_valid: torch.Tensor) -> torch.Tensor:
    sel = (loc_valid > 0) & (loc.sum(dim=1) > 0)
    if not sel.any():
        return loc_logits.new_zeros(())
    logits = loc_logits[sel].masked_fill(~ev_mask[sel], -1e4)
    target = loc[sel] / loc[sel].sum(dim=1, keepdim=True)
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def unified_loss(model, out: dict, batch, weights: dict, stage: int = 2,
                 agreement_min_reliability: float = 0.75) -> tuple[torch.Tensor, dict]:
    parts: dict = {}
    total = out["rho"].new_zeros(())
    w = {k: float(v) for k, v in weights.items()}
    ym, yc = batch.y_m, batch.y_c
    vm, vc = ym >= 0, yc >= 0
    # branch-level auxiliary objectives (stage 1, and kept in stage 2)
    for y, valid, target in (("m", vm, ym), ("c", vc, yc)):
        if y not in model.tasks or not valid.any():
            continue
        for k, a in (("T", out["a_T"]), ("G", out["a_G"])):
            sel = valid & (a > 0)
            if f"aux_{y}{k}" in out and sel.any():
                l_aux = F.cross_entropy(out[f"aux_{y}{k}"][sel], target[sel])
                parts[f"aux_{y}{k}"] = float(l_aux.detach())
                total = total + (1.0 if stage == 1 else w.get("auxiliary", 0.2)) * l_aux
    if "loc_logits" in out:
        l_loc = localization_loss(out["loc_logits"], batch.loc, batch.ev_mask, batch.loc_valid)
        parts["loc"] = float(l_loc.detach())
        total = total + w.get("localization", 0.5) * l_loc
    if stage == 1:
        return total, parts
    if "logits_m" in out and vm.any():
        l_m = F.cross_entropy(out["logits_m"][vm], ym[vm])
        parts["malware"] = float(l_m.detach())
        total = total + w.get("malware", 1.0) * l_m
        l_cal = dca_loss(out["logits_m"][vm], ym[vm])
        parts["cal_m"] = float(l_cal.detach())
        total = total + w.get("calibration", 0.05) * l_cal
    if "logits_c" in out and vc.any():
        l_c = F.cross_entropy(out["logits_c"][vc], yc[vc])
        parts["custody"] = float(l_c.detach())
        total = total + w.get("custody", 1.0) * l_c
        l_cal = dca_loss(out["logits_c"][vc], yc[vc])
        parts["cal_c"] = float(l_cal.detach())
        total = total + w.get("calibration", 0.05) * l_cal
    # reliable cross-view agreement
    both = (out["a_T"] > 0) & (out["a_G"] > 0) & (out["rho_T"] >= agreement_min_reliability) \
        & (out["rho_G"] >= agreement_min_reliability)
    if both.any() and w.get("agreement", 0.0) > 0 and any(k.startswith("aux_") for k in out):
        l_ag = out["rho"].new_zeros(())
        for y in model.tasks:
            if f"aux_{y}T" not in out or f"aux_{y}G" not in out:
                continue
            pt = torch.softmax(out[f"aux_{y}T"][both], -1)
            pg = torch.softmax(out[f"aux_{y}G"][both], -1)
            l_ag = l_ag + js_divergence(pt, pg).mean()
        parts["agree"] = float(l_ag.detach())
        total = total + w["agreement"] * l_ag
    if w.get("regularization", 0.0) > 0:
        reg = sum((p ** 2).sum() for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1)
        total = total + w["regularization"] * reg
    return total, parts
