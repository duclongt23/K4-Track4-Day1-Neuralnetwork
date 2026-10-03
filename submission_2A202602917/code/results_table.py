"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    path = Path(results_dir) / f"{exp_id}.json"
    
    data_to_save = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"]
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)
        
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    dir_path = Path(results_dir)
    if not dir_path.exists():
        return []
        
    results = []
    for p in dir_path.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            res = json.load(f)
            results.append(res)
            
    results.sort(key=lambda x: x["cfg"]["exp_id"])
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file."""
    cfg = result["cfg"]
    summary = result["summary"]
    
    row = {
        "exp_id": cfg["exp_id"],
        "group": cfg.get("group", ""),
        "description": cfg.get("description", ""),
        "loss": cfg["loss"],
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "hidden": str(cfg["hidden"]),
        "dropout": cfg["dropout"],
        "clip_norm": cfg["clip_norm"] if cfg["clip_norm"] is not None else "",
        "precision": cfg["precision"],
        "init": cfg["init"],
        "seed": cfg["seed"],
        
        "step0_loss": summary["step0_loss"],
        "best_val_loss": summary["best_val_loss"],
        "best_epoch": summary["best_epoch"],
        "final_train_loss": summary.get("final_train_loss", ""),
        "final_val_loss": summary.get("final_val_loss", ""),
        "val_acc": summary["val_acc"],
        "val_macro_f1": summary["val_macro_f1"],
        "time_per_epoch_s": summary["time_per_epoch_s"],
        "peak_mem_MB": summary["peak_mem_MB"],
        "diverged": summary["diverged"],
        
        "eval_acc": eval_scores["accuracy"] if eval_scores else "",
        "eval_macro_f1": eval_scores["macro_f1"] if eval_scores else "",
        "figure_file": f"figures/{cfg['exp_id']}.png",
        "notes": notes
    }
    
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path."""
    import openpyxl
    
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    
    # Read headers
    headers = {}
    for col_idx, cell in enumerate(ws[1], start=1):
        if cell.value:
            headers[cell.value] = col_idx
            
    # Write rows
    start_row = 2
    for row_idx, row_data in enumerate(rows, start=start_row):
        for key, value in row_data.items():
            if key in headers:
                col_idx = headers[key]
                ws.cell(row=row_idx, column=col_idx, value=value)
                
    wb.save(out_path)
