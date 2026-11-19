"""Train the frozen MNIST digit classifier (digits 1..9, black on white).

Used by the codec's perceptual losses and to score decoded MNIST Sudokus.

    python -m nca.extensions.spatial_reasoning.scripts.train_digit_classifier --out checkpoints/mnist_cnn.pt
"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ..codec import MNISTCNN
from .common import seed_everything


def digits_1_to_9(root, train, size):
    """Inverted MNIST digits 1..9 resized to ``size``, labels shifted to 0..8."""
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    mnist = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
    keep = (mnist.targets >= 1).nonzero().flatten().tolist()
    images = torch.stack([1.0 - mnist[i][0] for i in keep])
    return TensorDataset(images, mnist.targets[keep] - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="checkpoints/mnist_cnn.pt")
    parser.add_argument("--dataroot", default="datasets")
    parser.add_argument("--size", type=int, default=14, help="Input resolution; cells are resized to it.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    train = DataLoader(digits_1_to_9(args.dataroot, True, args.size), batch_size=args.batch_size, shuffle=True)
    test = DataLoader(digits_1_to_9(args.dataroot, False, args.size), batch_size=args.batch_size)
    model = MNISTCNN(num_classes=9).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train:
            loss = nn.functional.cross_entropy(model(images.to(args.device)), labels.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(args.device)).argmax(1).cpu() == y).sum().item() for x, y in test)
        print(f"Epoch {epoch}/{args.epochs}: test accuracy {correct / len(test.dataset):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": 9, "input_size": args.size}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
