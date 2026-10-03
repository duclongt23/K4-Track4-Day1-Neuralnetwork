"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    f1_scores = []
    for c in range(cm.shape[0]):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        
        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        f1_scores.append(F1)
        
    return float(np.mean(f1_scores))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        preds.append(pred)
    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total_loss = 0.0
    preds = []
    N = len(X)
    
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        
        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_onehot = F.one_hot(yb, num_classes=7).float()
            loss = F.mse_loss(logits, y_onehot, reduction="sum")
            
        total_loss += loss.item()
        preds.append(logits.argmax(dim=1))
        
    preds = torch.cat(preds)
    avg_loss = total_loss / N
    acc = (preds == y).float().mean().item()
    
    # Ma trận nhầm lẫn
    preds_np = preds.cpu().numpy()
    y_np = y.cpu().numpy()
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(y_np, preds_np):
        cm[t, p] += 1
        
    macro_f1 = macro_f1_from_confusion(cm)
    
    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Loss {loss_name} not supported")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    
    device = data["X_tr"].device
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS.get(cfg["hidden"], count_params(model))
    
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], 
                                weight_decay=cfg["weight_decay"], momentum=cfg.get("momentum", 0.9))
    
    scaler = None
    if cfg["precision"] == "fp16":
        scaler = torch.amp.GradScaler('cuda')
        
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])
    
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    # Train subset for evaluating train loss quickly
    subset_size = min(50000, len(X_tr))
    X_tr_sub, y_tr_sub = X_tr[:subset_size], y_tr[:subset_size]
    
    step0_res = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]
    
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], 
        "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []
    }
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    diverged = False
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_grad_norms = []
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=generator, shuffle=True):
            if cfg["precision"] == "fp16":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            elif cfg["precision"] == "bf16":
                with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break
                
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                epoch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                epoch_grad_norms.append(gn)
                optimizer.step()
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        
        if diverged:
            print(f"Epoch {epoch}: Diverged (NaN/Inf loss). Stopping.")
            break
            
        train_res = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg["loss"])
        val_res = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
        
        avg_gn = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)
        
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            import copy
            best_state = copy.deepcopy(model.state_dict())
            
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": history["val_acc"][best_epoch-1] if best_epoch > 0 else None,
        "val_macro_f1": history["val_macro_f1"][best_epoch-1] if best_epoch > 0 else None,
        "time_per_epoch_s": np.mean(history["epoch_time_s"]) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        "diverged": diverged
    }
    
    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    import pandas as pd
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    
    import subprocess
    import sys
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Running evaluation script for predictions at: {pred_path}")
    subprocess.run([sys.executable, f"{repo_root}/scripts/evaluate.py", "--pred", pred_path])
