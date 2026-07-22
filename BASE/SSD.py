from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import BASE.base_utils as base


def diagonal_importance(
    model: nn.Module,
    x,
    y,
    device: torch.device,
    batch_size: int,
    pos_weight: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    importance = {name: torch.zeros_like(param, device="cpu") for name, param in model.named_parameters()}
    seen = 0

    model.eval()
    for start, end in base.batch_slices(len(x), batch_size):
        xb = torch.from_numpy(x[start:end]).to(device)
        yb = torch.from_numpy(y[start:end]).to(device)
        model.zero_grad(set_to_none=True)
        criterion(model(xb), yb).backward()
        batch_n = end - start
        seen += batch_n
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += param.grad.detach().pow(2).cpu() * batch_n

    model.zero_grad(set_to_none=True)
    if seen == 0:
        raise ValueError("Cannot estimate SSD importance on an empty dataset.")
    return {name: value / seen for name, value in importance.items()}


def apply_ssd(
    model: nn.Module,
    forget_importance: dict[str, torch.Tensor],
    retain_importance: dict[str, torch.Tensor],
    alpha: float,
    damping: float,
    min_scale: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    changed = 0
    total = 0
    min_seen = 1.0

    with torch.no_grad():
        for name, param in model.named_parameters():
            f_imp = forget_importance[name].to(param.device)
            r_imp = retain_importance[name].to(param.device)
            mask = f_imp > alpha * (r_imp + eps)
            total += mask.numel()
            changed += int(mask.sum().item())

            scale = torch.ones_like(param)
            scale[mask] = torch.clamp(
                damping * (r_imp[mask] + eps) / (f_imp[mask] + eps),
                min=min_scale,
                max=1.0,
            )
            if mask.any():
                min_seen = min(min_seen, float(scale[mask].min().item()))
            param.mul_(scale)

    return {"ssd_changed_fraction": changed / max(1, total), "ssd_min_scale": min_seen}


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train, forget = base.load_data(ROOT / "data")
    feature_cols, target_cols = base.get_columns(train)
    retain, validation, validation_ids = base.build_splits(
        train,
        forget,
        target_cols,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    x_forget, y_forget = base.frame_to_arrays(forget, feature_cols, target_cols)
    x_retain, y_retain = base.frame_to_arrays(retain, feature_cols, target_cols)
    x_validation, y_validation = base.frame_to_arrays(validation, feature_cols, target_cols)
    x_ret_imp, y_ret_imp = base.sample_arrays(x_retain, y_retain, args.retain_importance_rows, args.seed)
    x_for_imp, y_for_imp = base.sample_arrays(x_forget, y_forget, args.forget_importance_rows, args.seed + 1)

    payload, model = base.load_artifact(ROOT / "data" / "model_artifact", device)
    pos_weight = base.compute_pos_weight(y_retain, device) if args.pos_weight else None

    start = time.perf_counter()
    retain_importance = diagonal_importance(model, x_ret_imp, y_ret_imp, device, args.batch_size, pos_weight)
    forget_importance = diagonal_importance(model, x_for_imp, y_for_imp, device, args.batch_size, pos_weight)
    stats = apply_ssd(model, forget_importance, retain_importance, args.alpha, args.damping, args.min_scale)
    elapsed = time.perf_counter() - start

    metrics = base.metric_bundle(
        y_forget,
        base.predict_logits(model, x_forget, device, args.eval_batch_size),
        y_validation,
        base.predict_logits(model, x_validation, device, args.eval_batch_size),
    )
    output_dir = ROOT / "submissions" / f"TIMmate_SSD_{args.version}"
    base.save_submission(payload, base.cpu_state_dict(model), validation_ids, output_dir, elapsed)

    print({"output_dir": str(output_dir), "execution_time": elapsed, **stats, **metrics})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSD-only TIMmate submission.")
    parser.add_argument("--version", default="V1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-size", type=int, default=12000)
    parser.add_argument("--retain-importance-rows", type=int, default=30000)
    parser.add_argument("--forget-importance-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--damping", type=float, default=0.9)
    parser.add_argument("--min-scale", type=float, default=0.7)
    parser.add_argument("--no-pos-weight", dest="pos_weight", action="store_false")
    parser.set_defaults(pos_weight=True)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
