"""Tab. 2: MNIST Sudoku with the frozen codec around an array-trained NCA.

The encoder reads the handwritten clues into a one-hot board, the NCA solves
it in board space and the decoder renders the result; clue cells keep their
input pixels. ASR/ACP are scored by the frozen digit classifier on the decoded
image (end-to-end); board-space numbers are printed for reference.

    python -m nca.extensions.spatial_reasoning.scripts.eval_mnist_sudoku \
        --checkpoint train_log/<sudoku run>/ca_final.pt \
        --codec-config nca/extensions/spatial_reasoning/configs/codec.yaml \
        --encoder train_log/<encoder run>/encoder_final.pt \
        --decoder train_log/<decoder run>/decoder_final.pt
"""

import argparse

import torch
import torch.nn.functional as F

from nca.utils.config import load_config

from ..codec import DigitClassifier, PixelSudokuBoardEncoder, SudokuBoardDecoder
from ..config import settings
from ..data.factories import dataroot
from ..data.mnist_sudoku import MNISTSudokuDataset
from ..data.sudoku import GRID_SIZE
from ..evaluation import rollout, sudoku_metrics, summarize
from .common import add_checkpoint_args, load_run, print_table, seed_everything
from .eval_sudoku import BUCKETS


def load_codec(codec_config, encoder_path, decoder_path, device):
    """Frozen encoder, decoder and digit classifier described by a codec config."""
    cfg = settings(codec_config).CODEC
    size = cfg.CELL_SIZE
    encoder = PixelSudokuBoardEncoder(
        cell_size=size, base_channels=cfg.ENCODER_BASE_CHANNELS, hidden_dim=cfg.ENCODER_HIDDEN_DIM,
    )
    decoder = SudokuBoardDecoder(
        cell_size=size, base_channels=cfg.DECODER_BASE_CHANNELS,
        hidden_dim=cfg.DECODER_HIDDEN_DIM, style_dim=cfg.DECODER_STYLE_DIM,
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    classifier = DigitClassifier(cfg.CLASSIFIER_PATH)
    return encoder.eval().to(device), decoder.eval().to(device), classifier.to(device)


def one_hot_board(logits):
    return F.one_hot(logits.argmax(dim=1), GRID_SIZE).permute(0, 3, 1, 2).float()


@torch.no_grad()
def evaluate_bucket(model, codec, dataset, n_puzzles, n_steps, batch_size, channel_n, device):
    encoder, decoder, classifier = codec
    board_results, pixel_results = {}, {}
    for start in range(0, n_puzzles, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n_puzzles))]
        batch = {key: torch.stack([item[key] for item in items]).to(device) for key in items[0]}
        given_board, given_pixels = batch["given_mask_board"], batch["given_mask_pixels"]

        state = torch.zeros(len(items), channel_n, GRID_SIZE, GRID_SIZE, device=device)
        state[:, :GRID_SIZE] = one_hot_board(encoder(batch["seed_pixels"])) * given_board
        state[:, GRID_SIZE:] = torch.randn_like(state[:, GRID_SIZE:]) * 0.1
        logits = rollout(model, state, given_board, n_steps)[:, :GRID_SIZE]

        decoded, _ = decoder(one_hot_board(logits))
        pixels = given_pixels * batch["seed_pixels"] + (1.0 - given_pixels) * decoded
        pixel_digits, _ = classifier.classify_grid(pixels, cell_size=dataset.cell_size)

        target = batch["target_board"].argmax(dim=1)
        for results, digits in ((board_results, logits.argmax(dim=1)), (pixel_results, pixel_digits)):
            for key, value in sudoku_metrics(digits, target, given_board[:, 0]).items():
                results.setdefault(key, []).append(value.cpu())
    return summarize(board_results), summarize(pixel_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_args(parser, default_steps=1500)
    parser.add_argument("--codec-config", required=True, help="Codec YAML with an EXTENSIONS.SPATIAL_REASONING.CODEC section.")
    parser.add_argument("--encoder", required=True, help="Encoder checkpoint.")
    parser.add_argument("--decoder", required=True, help="Decoder checkpoint.")
    parser.add_argument("--n-puzzles", type=int, default=1000)
    parser.add_argument("--buckets", nargs="+", default=list(BUCKETS), choices=list(BUCKETS))
    parser.add_argument("--seed", type=int, default=None, help="Test-split seed; default SEED + 10000 of the NCA run.")
    args = parser.parse_args()

    config, model = load_run(args, cond_dim=1, size=GRID_SIZE)
    codec_config = load_config(args.codec_config)
    codec = load_codec(codec_config, args.encoder, args.decoder, config.DEVICE)
    seed = args.seed if args.seed is not None else (config.SEED + 10_000 if config.SEED != -1 else 99_999)

    rows = []
    for name in args.buckets:
        seed_everything(seed)
        dataset = MNISTSudokuDataset(
            BUCKETS[name], args.n_puzzles, train=False, seed=seed,
            cell_size=settings(codec_config).CODEC.CELL_SIZE, dataroot=dataroot(codec_config),
        )
        board, pixel = evaluate_bucket(
            model, codec, dataset, args.n_puzzles, args.n_steps, args.batch_size, config.MODEL.CHANNEL_N, config.DEVICE,
        )
        rows.append((
            name, f"{pixel['valid']:.2f}", f"{pixel['acp_valid']:.2f}", f"{board['valid']:.2f}", f"{board['acp_valid']:.2f}",
        ))
    print_table(
        f"{args.checkpoint} ({args.n_puzzles} puzzles, {args.n_steps} steps, seed {seed})",
        ("Bucket", "ASR", "ACP", "ASR(board)", "ACP(board)"),
        rows,
    )


if __name__ == "__main__":
    main()
