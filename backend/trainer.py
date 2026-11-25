# Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE.
"""Training and evaluation for the three task kinds (classification, regression, anomaly autoencoder)."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from . import archspec

torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # GPU workers train on CUDA


def split(y, task, seed=0, frac=(0.7, 0.15, 0.15), groups=None, mode="auto"):
    """Train/val/test indices.

    With recording groups (and mode "auto"), every recording goes to exactly one split, so no window of a test
    recording is ever seen in training, even when a recording holds several classes. Recordings are stratified by
    their majority class (regression: one stratum). A stratum with fewer than 3 recordings cannot be split by
    recording; its windows are split at window level and `split_mode` reports "mixed".
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = {"train": [], "val": [], "test": []}
    modes = []
    use_groups = groups is not None and mode != "window" and len(set(np.asarray(groups).tolist())) < n
    if use_groups:
        groups = np.asarray(groups)
        members: dict = {}
        for i, g in enumerate(groups.tolist()):
            members.setdefault(g, []).append(i)
        strata: dict = {}
        for g, ix in members.items():
            key = None if task == "regression" else int(np.bincount(np.asarray(y[ix], dtype=int)).argmax())
            strata.setdefault(key, []).append(g)
        for key, glist in strata.items():
            if len(glist) < 3:                         # too few recordings: window-level for this stratum
                ix = rng.permutation([i for g in glist for i in members[g]])
                a_, b_ = int(len(ix) * frac[0]), int(len(ix) * (frac[0] + frac[1]))
                idx["train"] += list(ix[:a_]); idx["val"] += list(ix[a_:b_]); idx["test"] += list(ix[b_:])
                modes.append("window")
                continue
            order = [glist[i] for i in rng.permutation(len(glist))]
            total = sum(len(members[g]) for g in order)
            acc, parts = 0, {"train": [], "val": [], "test": []}
            for i, g in enumerate(order):
                left = len(order) - i
                if not parts["test"] and left == 1:
                    dest = "test"
                elif not parts["val"] and left == 2 and parts["train"]:
                    dest = "val"
                else:
                    dest = "train" if acc < frac[0] * total else "val" if acc < (frac[0] + frac[1]) * total else "test"
                parts[dest].append(g)
                acc += len(members[g])
            for dest, gl in parts.items():
                for g in gl:
                    idx[dest] += members[g]
            modes.append("recording")
    else:
        keys = [None] if task == "regression" else list(np.unique(y))
        for k in keys:
            ix = rng.permutation(np.arange(n) if k is None else np.where(y == k)[0])
            a_, b_ = int(len(ix) * frac[0]), int(len(ix) * (frac[0] + frac[1]))
            idx["train"] += list(ix[:a_]); idx["val"] += list(ix[a_:b_]); idx["test"] += list(ix[b_:])
            modes.append("window")
    if task == "anomaly":                    # the autoencoder learns normal only; anomalies are for evaluation
        idx["train"] = [i for i in idx["train"] if y[i] == 0]
        idx["val"] = [i for i in idx["val"] if y[i] == 0]
    out = {k: np.array(sorted(v), dtype=int) for k, v in idx.items()}
    out["split_mode"] = "recording" if modes and all(m == "recording" for m in modes) else \
        "mixed" if "recording" in modes else "window"
    return out


def norm_stats(X):
    mean = X.mean(axis=(0, 2, 3), keepdims=True)
    std = X.std(axis=(0, 2, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)          # a constant channel carries no information: do not blow it up
    return mean.astype(np.float32), std.astype(np.float32)


def build_spec(layers, X, task, n_labels):
    n_out = n_labels if task == "classification" else 1 if task == "regression" else int(np.prod(X.shape[1:]))
    return archspec.normalize({"input": {"channels": X.shape[1], "height": X.shape[2], "width": X.shape[3]},
                               "layers": layers, "outputs": n_out})


def train(spec, Xn, y, idx, task, epochs=30, lr=5e-3, batch=None, min_steps=1500, log=lambda m: None, progress=lambda p: None,
          cancelled=lambda: False, y_stats=None, net=None, augment=None, class_weights=True):
    torch.manual_seed(0)
    net = (net if net is not None else archspec.SpecNet(spec)).to(DEVICE)
    n_train = len(idx["train"])
    batch = batch or int(min(64, max(16, n_train // 25)))
    # small datasets need more passes: at least `min_steps` optimizer updates in total
    epochs = max(epochs, math.ceil(min_steps / max(1, math.ceil(n_train / batch))))
    epochs = min(epochs, 300)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    Xt = torch.from_numpy(Xn).to(DEVICE)
    if task == "classification":
        counts = np.bincount(y[idx["train"]], minlength=spec["outputs"]).astype(np.float32)
        w = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(counts), 0), dtype=torch.float32)
        if not class_weights:
            w = torch.ones_like(w)
        loss_fn = nn.CrossEntropyLoss(weight=w.to(DEVICE))
        yt = torch.from_numpy(y.astype(np.int64)).to(DEVICE)
    elif task == "regression":
        loss_fn = nn.MSELoss()
        yt = torch.from_numpy(((y - y_stats[0]) / y_stats[1]).astype(np.float32))[:, None].to(DEVICE)
    else:
        loss_fn = nn.MSELoss()
        yt = Xt.flatten(1)
    best, best_state, bad = math.inf, None, 0
    history = []
    for ep in range(epochs):
        if cancelled():
            raise InterruptedError("cancelled")
        net.train()
        perm = np.random.default_rng(ep).permutation(idx["train"])
        tot = 0.0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            xb = Xt[b]
            if task != "anomaly":            # augmentation (training only): gain jitter, noise, time shift
                aug = augment if augment is not None else {"gain": 0.05, "noise": 0.0, "shift": 0}
                if aug.get("gain"):
                    xb = xb * (1 + aug["gain"] * torch.randn(len(b), 1, 1, 1, device=DEVICE))
                if aug.get("noise"):
                    xb = xb + aug["noise"] * torch.randn_like(xb)
                if aug.get("shift"):
                    xb = torch.roll(xb, int(np.random.default_rng(ep * 100003 + s).integers(-aug["shift"], aug["shift"] + 1)), dims=3)
            opt.zero_grad()
            loss = loss_fn(net(xb), yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xt[idx["val"]]), yt[idx["val"]]))
        if not math.isfinite(vl):
            log(f"  diverged at epoch {ep + 1}" + ("; keeping the best weights so far" if best_state else ""))
            break
        history.append({"epoch": ep + 1, "train_loss": round(tot / max(len(perm), 1), 5), "val_loss": round(vl, 5)})
        progress((ep + 1) / epochs)
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"  epoch {ep + 1}/{epochs} train {tot / max(len(perm), 1):.4f} val {vl:.4f}")
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= max(10, epochs // 5):
                log(f"  early stop at epoch {ep + 1}")
                break
    net = net.cpu()
    if best_state is None:
        raise ValueError("training produced no finite validation loss (check the data for extreme values, or use "
                         "more data / a smaller learning rate)")
    net.load_state_dict(best_state)
    net.eval()
    net.best_val = best
    return net, history


def init_from(net, state: dict) -> list[str]:
    """Fine-tuning start: copy every parameter whose name and shape match (the head is re-initialised when the
    customer's label count differs). Returns the copied names."""
    own = net.state_dict()
    copied = [k for k, v in state.items() if k in own and own[k].shape == v.shape]
    own.update({k: state[k] for k in copied})
    net.load_state_dict(own)
    return copied


def qat_finetune(net, act_q, Xn, y, idx, task, epochs=10, log=lambda m: None, cancelled=lambda: False, y_stats=None):
    """Quantization-aware fine-tuning on the BN-folded network with the calibrated activation ranges fixed and
    weights fake-quantized per channel, the way the exporter and SAI's INT8 kernels will compute."""
    net.set_qat(act_q)
    try:
        net, hist = train(net.spec, Xn, y, idx, task, epochs=epochs, lr=5e-4, min_steps=300, log=log,
                          cancelled=cancelled, y_stats=y_stats, net=net)
    finally:
        net.set_qat(None)
    return net, hist


def metrics(task, y_true, out, labels, y_stats=None, x_flat=None, threshold=None):
    """out: float outputs [N, n_out] (dequantized for INT8 runs)."""
    if task == "classification":
        pred = out.argmax(1)
        k = len(labels)
        cm = np.zeros((k, k), int)
        for t, p in zip(y_true, pred):
            cm[int(t), int(p)] += 1
        rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
        prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        return {"accuracy": round(float((pred == y_true).mean()), 4), "macro_f1": round(float(f1.mean()), 4),
                "confusion": cm.tolist(), "per_class_recall": [round(float(r), 3) for r in rec],
                "headline": "accuracy", "value": round(float((pred == y_true).mean()), 4)}
    if task == "regression":
        pred = out[:, 0] * y_stats[1] + y_stats[0]
        err = pred - y_true
        mae = float(np.abs(err).mean())
        r2 = 1 - float((err ** 2).sum() / max(((y_true - y_true.mean()) ** 2).sum(), 1e-9))
        return {"mae": round(mae, 4), "rmse": round(float(np.sqrt((err ** 2).mean())), 4), "r2": round(r2, 4),
                "headline": "r2", "value": round(r2, 4)}
    err = ((out - x_flat) ** 2).mean(1)
    auc = _auc(err, y_true)
    flag = err > threshold
    tpr = float(flag[y_true == 1].mean()) if (y_true == 1).any() else None
    fpr = float(flag[y_true == 0].mean()) if (y_true == 0).any() else None
    auc = None if auc is None else round(auc, 4)
    return {"auc": auc, "threshold": round(float(threshold), 6),
            "detection_rate": None if tpr is None else round(tpr, 4),
            "false_alarm_rate": None if fpr is None else round(fpr, 4), "headline": "auc", "value": auc,
            "note": None if auc is not None else "no anomalies in the test split: AUC undefined"}


def _auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
