"""Train the MNIST Sudoku encoder or decoder (``EXTENSIONS.SPATIAL_REASONING.CODEC.MODE``).

Encoder: masked cross-entropy on clue cells. Decoder: conditional VAE with
    w_KL KL + w_pix L1 + w_feat feature-L1 + w_cls classifier-CE + w_TV TV + w_ink ink + w_bin binary
on occupied cells; perceptual terms use the frozen digit classifier.
Checkpoints go to ``train_log/<TRAIN_NAME>_<MODE>_<timestamp>/``.

    python -m nca.extensions.spatial_reasoning.scripts.train_codec --config nca/extensions/spatial_reasoning/configs/codec.yaml \
        -o EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=encoder
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nca.training.training_utils import create_warmup_cosine_scheduler
from nca.utils.config import load_config
from scripts.train_ca import apply_overrides, parse_override_strings

from ..codec import DigitClassifier, PixelSudokuBoardEncoder, SudokuBoardDecoder, extract_sudoku_cells
from ..config import settings
from ..data.factories import dataroot
from ..data.mnist_sudoku import MNISTSudokuDataset
from .common import seed_everything


def encoder_losses(batch, encoder):
    """Masked cross-entropy between encoder logits and the clue digits."""
    logits = encoder(batch["seed_pixels"])
    mask = batch["given_mask_board"][:, 0]
    digits = batch["seed_board"].argmax(dim=1)
    ce = F.cross_entropy(logits, digits, reduction="none")
    loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
    accuracy = ((logits.argmax(dim=1) == digits).float() * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"board_ce": loss, "board_acc": accuracy}


def decoder_losses(batch, decoder, classifier, cfg, cell_size):
    """Composite decoder loss on clue boards or, with ``SOLUTION_RATIO``, full solutions."""
    use_solution = (torch.rand(batch["seed_board"].shape[0], device=batch["seed_board"].device) < cfg.SOLUTION_RATIO)
    use_solution = use_solution.view(-1, 1, 1, 1)
    board = torch.where(use_solution, batch["target_board"], batch["seed_board"])
    target = torch.where(use_solution, batch["target_pixels"], batch["seed_pixels"])
    rendered, kl = decoder(board, target_pixels=target)

    occupied = board.sum(dim=1, keepdim=True) > 0
    mask = occupied.float().repeat_interleave(cell_size, dim=2).repeat_interleave(cell_size, dim=3)
    n_mask = mask.sum().clamp_min(1.0)
    ink = (1.0 - rendered) * mask
    pixel_l1 = ((rendered - target).abs() * mask).sum() / n_mask
    tv = (ink[:, :, 1:] - ink[:, :, :-1]).abs().mean() + (ink[..., 1:] - ink[..., :-1]).abs().mean()
    ink_loss = ink.sum() / n_mask
    binary = (rendered * (1.0 - rendered) * mask).sum() / n_mask

    cells = occupied.reshape(-1)
    rendered_cells = extract_sudoku_cells(rendered, 9, cell_size)[cells]
    target_cells = extract_sudoku_cells(target, 9, cell_size)[cells]
    with torch.no_grad():
        target_features = classifier.features(target_cells)
    feature_l1 = F.l1_loss(classifier.features(rendered_cells), target_features)
    logits = classifier(rendered_cells)
    digits = board.argmax(dim=1).reshape(-1)[cells]
    classifier_ce = F.cross_entropy(logits.float(), digits)

    loss = (
        cfg.KL_WEIGHT * kl + cfg.PIXEL_L1_WEIGHT * pixel_l1 + cfg.FEATURE_L1_WEIGHT * feature_l1
        + cfg.CLASSIFIER_CE_WEIGHT * classifier_ce + cfg.TV_WEIGHT * tv + cfg.INK_WEIGHT * ink_loss
        + cfg.BINARY_WEIGHT * binary
    )
    return loss, {
        "kl": kl, "pixel_l1": pixel_l1, "feature_l1": feature_l1, "classifier_ce": classifier_ce,
        "classifier_acc": (logits.argmax(dim=1) == digits).float().mean(), "tv": tv, "ink": ink_loss,
        "binary": binary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("-o", "--override", action="append", default=[], help="KEY=VALUE, e.g. EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=encoder")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), parse_override_strings(args.override))
    section, device = settings(config), config.DEVICE
    cfg = section.CODEC
    if cfg is None:
        raise ValueError(f"{args.config} has no EXTENSIONS.SPATIAL_REASONING.CODEC section.")
    if config.SEED != -1:
        seed_everything(config.SEED)

    out = Path("train_log") / f"{config.LOGGING.TRAIN_NAME}_{cfg.MODE}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True), sort_keys=False))

    cell_size = cfg.CELL_SIZE
    dataset = MNISTSudokuDataset(
        section.SUDOKU.DIFFICULTY, section.SUDOKU.TRAIN_SAMPLES, train=True,
        seed=config.SEED, cell_size=cell_size, dataroot=dataroot(config),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=True, drop_last=config.DATASET.DROP_LAST_BATCH,
    )

    if cfg.MODE == "encoder":
        model = PixelSudokuBoardEncoder(
            cell_size=cell_size, base_channels=cfg.ENCODER_BASE_CHANNELS, hidden_dim=cfg.ENCODER_HIDDEN_DIM,
        ).to(device)
        step_losses = lambda batch: encoder_losses(batch, model)
    else:
        model = SudokuBoardDecoder(
            cell_size=cell_size, base_channels=cfg.DECODER_BASE_CHANNELS,
            hidden_dim=cfg.DECODER_HIDDEN_DIM, style_dim=cfg.DECODER_STYLE_DIM,
        ).to(device)
        classifier = DigitClassifier(cfg.CLASSIFIER_PATH).to(device)
        step_losses = lambda batch: decoder_losses(batch, model, classifier, cfg, cell_size)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = create_warmup_cosine_scheduler(optimizer, cfg.WARMUP_STEPS, cfg.STEPS)
    if config.LOGGING.WANDB:
        import wandb
        wandb.init(project=config.LOGGING.PROJECT_NAME, name=out.name, config=cfg.model_dump())

    batches, running = iter(loader), {}
    model.train()
    for step in tqdm(range(1, cfg.STEPS + 1), desc=f"Training codec {cfg.MODE}", mininterval=1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        loss, metrics = step_losses(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics = {"loss": loss.item(), **{k: float(v) for k, v in metrics.items()}}
        if config.LOGGING.WANDB:
            wandb.log(metrics | {"lr": scheduler.get_last_lr()[0]}, step=step)
        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + value
        if step % cfg.LOG_INTERVAL == 0:
            print(f"Step {step}: " + ", ".join(f"{k}: {v / cfg.LOG_INTERVAL:.4f}" for k, v in running.items()))
            running = {}
        if step % cfg.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), out / f"{cfg.MODE}_{step}.pt")

    torch.save(model.state_dict(), out / f"{cfg.MODE}_final.pt")
    print(f"Saved {out / f'{cfg.MODE}_final.pt'}")


if __name__ == "__main__":
    main()
