"""
Inference script for NCA models.

Loads a checkpoint from a train_log directory, evolves the NCA step-by-step,
colors every frame using the dataset's ``batch_to_rgb`` (same as training logs),
and saves a video.

The output layout per frame mirrors the training logger exactly:
    Seed  |  Step k  |  Target
(one column per batch sample, row labels on the left)

Usage examples
--------------
python train_scripts/inference.py --log_dir train_logs/maze_20240101_120000
python train_scripts/inference.py --log_dir train_logs/maze_20240101_120000 \\
    --iter_n 64 --batch_size 4 --fps 15
python train_scripts/inference.py --log_dir train_logs/maze_20240101_120000 \\
    --checkpoint ca_final.pt --output results/maze.mp4
"""

import os
import sys
import argparse
import glob

import torch
import numpy as np
import imageio
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nca.utils.config import load_config
from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.data.datasets.base_dataset import NCADataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(log_dir: str, name: str | None) -> str:
    if name is not None:
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    final = os.path.join(log_dir, "ca_final.pt")
    if os.path.isfile(final):
        return final

    candidates = sorted(glob.glob(os.path.join(log_dir, "ca_*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {log_dir}")

    def _step(p):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base.split("_")[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def _make_frame(
    rows: list[tuple[str, torch.Tensor]],
    batch_size: int,
    label_width: int = 80,
    padding: int = 4,
) -> np.ndarray:
    """
    Compose a video frame from labeled RGB rows.

    Args:
        rows       : list of (label, rgb_tensor) where rgb_tensor is (B, 3, H, W)
        batch_size : how many samples per row
        label_width: pixel width of the left label strip
        padding    : padding between images in the grid

    Returns:
        (H, W, 3) uint8 numpy array
    """
    if not rows:
        raise ValueError("rows is empty")

    # Stack all row tensors and build grid
    stacked = torch.cat([rgb[:batch_size] for _, rgb in rows], dim=0)
    grid = vutils.make_grid(
        stacked.clamp(0, 1),
        nrow=batch_size,
        padding=padding,
        normalize=False,
        pad_value=1.0,
    )
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    grid_h, grid_w = grid_np.shape[:2]

    # Add label strip on the left
    full = np.full((grid_h, grid_w + label_width, 3), 255, dtype=np.uint8)
    full[:, label_width:] = grid_np

    pil = Image.fromarray(full)
    draw = ImageDraw.Draw(pil)

    n_rows = len(rows)
    row_h = grid_h // n_rows
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (label, _) in enumerate(rows):
        y_center = i * row_h + row_h // 2
        draw.text((4, y_center - 8), label, fill=(0, 0, 0), font=font)

    return np.array(pil)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="NCA inference + video export")
    parser.add_argument(
        "--log_dir", required=True,
        help="Train log directory (must contain config.yaml and a .pt checkpoint).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint filename inside --log_dir (default: auto-select latest).",
    )
    parser.add_argument(
        "--iter_n", type=int, default=None,
        help="NCA steps to run (default: ITER_N_MAX from config).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of samples to visualise (default: 8).",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second (default: 10).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video path (default: <log_dir>/inference.mp4).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (default: from config).",
    )
    parser.add_argument(
        "--train_split", action="store_true",
        help="Sample from the training split instead of validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir

    # --- Config ---
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found in {log_dir}")
    config = load_config(config_path)

    if args.device:
        config.DEVICE = args.device
    device = config.DEVICE

    iter_n = args.iter_n if args.iter_n is not None else config.TRAINING.ITER_N_MAX
    output_path = args.output or os.path.join(log_dir, "inference.mp4")

    print(f"Config     : {config_path}")
    print(f"Device     : {device}")
    print(f"Iterations : {iter_n}")
    print(f"Batch size : {args.batch_size}")

    # --- Dataset ---
    config.TRAINING.BATCH_SIZE = args.batch_size
    use_train = args.train_split
    dataloader, cond_dim, im_height, im_width = create_dataset(config, train=use_train)
    dataset = dataloader.get_dataset()

    seed, cond, target = next(iter(dataloader))
    seed   = seed[:args.batch_size].to(device)
    # Only use condition if model actually expects it (cond_dim > 0)
    cond   = cond[:args.batch_size].to(device) if cond is not None and cond_dim > 0 else None
    target = target[:args.batch_size].to(device)

    # --- Model ---
    model = create_model(config, cond_dim, im_height, im_width)
    ckpt_path = _find_checkpoint(log_dir, args.checkpoint)
    print(f"Checkpoint : {ckpt_path}")

    # Determine freeze_channels if model outputs fewer channels than state
    channel_n = config.MODEL.CHANNEL_N
    channel_out = getattr(config.MODEL, 'CHANNEL_OUT', channel_n)
    if channel_out < channel_n:
        # Freeze the first (channel_n - channel_out) channels
        freeze_channels = channel_n - channel_out - 1
        print(f"Freezing channels 0-{freeze_channels} (model outputs {channel_out} of {channel_n} channels)")
    else:
        freeze_channels = None

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Pre-color fixed tensors (seed and target never change) ---
    has_coloring = isinstance(dataset, NCADataset)
    if has_coloring:
        print("Using dataset.batch_to_rgb for coloring.")

    x0_cpu     = seed.detach().cpu()
    target_cpu = target.detach().cpu()
    cond_cpu   = cond.detach().cpu() if cond is not None else None

    def _rgb(t):
        """Convert tensor to RGB (B, 3, H, W) by taking first 3 channels."""
        c = t.shape[1]
        if c == 1:
            return t.repeat(1, 3, 1, 1).clamp(0, 1)
        return t[:, :3].clamp(0, 1)

    def colorize_state(x_cpu):
        """Return (x0_rgb, x_rgb, target_rgb) — all (B, 3, H, W)."""
        if has_coloring:
            x0_out, x_out, target_out = dataset.batch_to_rgb(x0_cpu, x_cpu, target_cpu, cond_cpu)
            # Ensure outputs are actually RGB (3 channels) - some datasets pass through unchanged
            if x0_out.shape[1] != 3:
                x0_out = _rgb(x0_out)
            if x_out.shape[1] != 3:
                x_out = _rgb(x_out)
            if target_out.shape[1] != 3:
                target_out = _rgb(target_out)
            return x0_out, x_out, target_out
        return _rgb(x0_cpu), _rgb(x_cpu), _rgb(target_cpu)

    # Seed and target RGB are constant — compute once
    x0_rgb, _, target_rgb = colorize_state(x0_cpu)

    # --- Evolve & collect frames ---
    video_frames = []
    state = seed.clone()

    with torch.no_grad():
        # Frame 0 = initial seed
        _, x_rgb, _ = colorize_state(x0_cpu)
        video_frames.append(_make_frame(
            [("Seed", x0_rgb), ("Step 0", x_rgb), ("Target", target_rgb)],
            batch_size=args.batch_size,
        ))

        for step in range(1, iter_n + 1):
            state = model(state, cond, freeze_channels=freeze_channels)
            _, x_rgb, _ = colorize_state(state.detach().cpu())
            video_frames.append(_make_frame(
                [("Seed", x0_rgb), (f"Step {step}", x_rgb), ("Target", target_rgb)],
                batch_size=args.batch_size,
            ))
            if step % max(1, iter_n // 10) == 0:
                print(f"  step {step:4d}/{iter_n}")

    # --- Save video ---
    # Ensure all frames have the same size (they should, but PIL resize as safety net)
    h, w = video_frames[0].shape[:2]
    video_frames = [
        f if f.shape[:2] == (h, w)
        else np.array(Image.fromarray(f).resize((w, h)))
        for f in video_frames
    ]

    print(f"Saving {len(video_frames)} frames → {output_path}")
    with imageio.get_writer(output_path, fps=args.fps, macro_block_size=1) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    print("Done.")


if __name__ == "__main__":
    main()
