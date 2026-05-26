import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.data.dataset import PairedRestoreDataset, build_train_val_splits
from src.infer import run_inference
from src.models.promptir import PromptIR
from src.utils.losses import SSIMLoss, charbonnier_loss, frequency_loss
from src.utils.metrics import psnr


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        msd = model.state_dict()
        merged = {k: (self.shadow[k] if k in self.shadow else v) for k, v in msd.items()}
        model.load_state_dict(merged)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, log_n_images: int = 3):
    model.eval()
    total_psnr = 0.0
    n = 0
    samples = []
    for deg, clean in loader:
        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred = model(deg).clamp(0.0, 1.0)
        batch_psnr = psnr(pred, clean).item()
        total_psnr += batch_psnr * deg.size(0)
        n += deg.size(0)
        if len(samples) < log_n_images:
            samples.append((deg[0].cpu(), pred[0].cpu(), clean[0].cpu()))
    return total_psnr / max(n, 1), samples


def _to_wandb_img(t: torch.Tensor):
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return wandb.Image(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="hw4_realse_dataset")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--test_dir", type=str, default="hw4_realse_dataset/test/degraded")
    parser.add_argument("--out_dir", type=str, default="./result")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag appended to run dir; if None uses timestamp")
    parser.add_argument("--infer_name", type=str, default=None, help="Filename (without dir) for auto-inference npz; default pred_{run_id}.npz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_per_type", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="nycu-dlcv-hw4-promptir")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_notes", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--use_fft", action="store_true")
    parser.add_argument("--use_gated", action="store_true")
    parser.add_argument("--char", action="store_true", help="Use Charbonnier loss instead of L1")
    parser.add_argument("--ssim_w", type=float, default=0.0, help="SSIM loss weight (0 disables)")
    parser.add_argument("--freq_w", type=float, default=0.0, help="Frequency loss weight (0 disables)")
    parser.add_argument("--use_ema", action="store_true", help="Track EMA of weights and use them for eval/inference")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--tta", action="store_true", help="Use x8 self-ensemble at inference time")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_tag is not None:
        run_id = args.run_tag
    elif args.wandb_run_name is not None:
        run_id = f"{timestamp}_{args.wandb_run_name}"
    else:
        run_id = timestamp
    ckpt_dir = Path(args.ckpt_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / (args.infer_name or f"pred_{run_id}.npz")
    if out_npz.exists():
        stamp = datetime.now().strftime("%H%M%S")
        out_npz = out_npz.with_name(f"{out_npz.stem}_{stamp}{out_npz.suffix}")
    print(f"run_id={run_id}\nckpt_dir={ckpt_dir}\nout_npz={out_npz}")

    train_items, val_items = build_train_val_splits(args.data_root, val_per_type=args.val_per_type)
    train_ds = PairedRestoreDataset(train_items, patch_size=args.patch_size, train=True)
    val_ds = PairedRestoreDataset(val_items, patch_size=None, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=True,
    )
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    model = PromptIR(
        decoder=True,
        use_se=args.use_se,
        use_fft=args.use_fft,
        use_gated=args.use_gated,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    ssim_loss = SSIMLoss().to(device) if args.ssim_w > 0 else None
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        config={**vars(args), "params_M": n_params / 1e6},
    )
    wandb.watch(model, log=None)

    best_psnr = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for deg, clean in pbar:
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(deg)
            if args.char:
                l_pix = charbonnier_loss(pred, clean)
                log_parts = {"train/charbonnier": l_pix.item()}
            else:
                l_pix = F.l1_loss(pred, clean)
                log_parts = {"train/l1": l_pix.item()}
            loss = l_pix
            if ssim_loss is not None:
                l_ssim = ssim_loss(pred.clamp(0, 1), clean)
                loss = loss + args.ssim_w * l_ssim
                log_parts["train/ssim"] = l_ssim.item()
            if args.freq_w > 0:
                l_freq = frequency_loss(pred, clean)
                loss = loss + args.freq_w * l_freq
                log_parts["train/frequency"] = l_freq.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            global_step += 1
            bs = deg.size(0)
            epoch_loss += loss.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log({"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], **log_parts}, step=global_step)

        scheduler.step()
        train_loss = epoch_loss / max(n_seen, 1)

        if ema is not None:
            eval_model = PromptIR(
                decoder=True,
                use_se=args.use_se,
                use_fft=args.use_fft,
                use_gated=args.use_gated,
            ).to(device)
            eval_model.load_state_dict(model.state_dict())
            ema.copy_to(eval_model)
        else:
            eval_model = model
        val_psnr, samples = evaluate(eval_model, val_loader, device, log_n_images=3)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_psnr={val_psnr:.3f}  time={elapsed:.1f}s")

        log = {
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            "val/psnr": val_psnr,
            "time/epoch_s": elapsed,
        }
        if samples:
            log["val/examples"] = [
                wandb.Image(
                    np.concatenate(
                        [
                            (deg.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                            (pred.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                            (clean.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                        ],
                        axis=1,
                    ),
                    caption=f"deg | pred | clean (epoch {epoch})",
                )
                for deg, pred, clean in samples
            ]
        wandb.log(log, step=global_step)

        state = {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "val_psnr": val_psnr,
            "args": vars(args),
        }
        torch.save(state, ckpt_dir / "last.pth")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(state, ckpt_dir / "best.pth")
            wandb.run.summary["best/val_psnr"] = best_psnr
            wandb.run.summary["best/epoch"] = epoch

    print(f"training done. best val_psnr={best_psnr:.3f}")

    # Auto-inference with best checkpoint (use EMA weights if available)
    best_state = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best_state["model"])
    if best_state.get("ema") is not None:
        merged = {k: best_state["ema"].get(k, v) for k, v in model.state_dict().items()}
        model.load_state_dict(merged)
        print("Loaded EMA weights for inference.")
    run_inference(model, args.test_dir, out_npz, device, tta=args.tta)

    if out_npz.exists():
        art = wandb.Artifact("pred", type="predictions")
        art.add_file(str(out_npz))
        wandb.log_artifact(art)

    wandb.finish()


if __name__ == "__main__":
    main()
