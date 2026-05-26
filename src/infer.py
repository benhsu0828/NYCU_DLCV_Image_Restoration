import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.promptir import PromptIR


_TTA_TRANSFORMS = [
    ("ident", lambda t: t, lambda t: t),
    ("hflip", lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
    ("vflip", lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
    ("hvflip", lambda t: torch.flip(t, dims=[-2, -1]), lambda t: torch.flip(t, dims=[-2, -1])),
    ("rot90", lambda t: torch.rot90(t, k=1, dims=[-2, -1]), lambda t: torch.rot90(t, k=-1, dims=[-2, -1])),
    ("rot180", lambda t: torch.rot90(t, k=2, dims=[-2, -1]), lambda t: torch.rot90(t, k=-2, dims=[-2, -1])),
    ("rot270", lambda t: torch.rot90(t, k=3, dims=[-2, -1]), lambda t: torch.rot90(t, k=-3, dims=[-2, -1])),
    ("trans", lambda t: t.transpose(-2, -1), lambda t: t.transpose(-2, -1)),
]


@torch.no_grad()
def tta_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    outs = []
    for _, fwd, inv in _TTA_TRANSFORMS:
        y = model(fwd(x))
        outs.append(inv(y))
    return torch.stack(outs, dim=0).mean(dim=0)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    test_dir: str | Path,
    out_path: str | Path,
    device: torch.device,
    tta: bool = False,
):
    model.eval()
    test_dir = Path(test_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: (len(p.stem), p.stem),
    )
    predict = tta_predict if tta else (lambda m, x: m(x))
    images_dict: dict[str, np.ndarray] = {}
    desc = "infer (TTA x8)" if tta else "infer"
    for p in tqdm(files, desc=desc):
        img = np.array(Image.open(p).convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        y = predict(model, x).clamp(0.0, 1.0).squeeze(0).cpu().numpy()
        y = (y * 255.0).round().astype(np.uint8)
        images_dict[p.name] = y

    np.savez(out_path, **images_dict)
    print(f"Saved {len(images_dict)} predictions to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth")
    parser.add_argument("--test_dir", type=str, default="hw4_realse_dataset/test/degraded")
    parser.add_argument("--out", type=str, default="pred.npz")
    parser.add_argument("--tta", action="store_true", help="Use x8 self-ensemble TTA")
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--use_fft", action="store_true")
    parser.add_argument("--use_gated", action="store_true")
    parser.add_argument("--use_ema_weights", action="store_true", help="Load EMA weights from ckpt if present")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(
        decoder=True,
        use_se=args.use_se,
        use_fft=args.use_fft,
        use_gated=args.use_gated,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    msd = state["model"] if "model" in state else state
    model.load_state_dict(msd)
    if args.use_ema_weights and isinstance(state, dict) and state.get("ema"):
        merged = {k: state["ema"].get(k, v) for k, v in model.state_dict().items()}
        model.load_state_dict(merged)
        print("Loaded EMA weights.")
    run_inference(model, args.test_dir, args.out, device, tta=args.tta)


if __name__ == "__main__":
    main()
