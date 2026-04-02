"""Dataset implementations for NCA-Torch"""

from nca.data.datasets.base_dataset import NCADataset
from nca.data.datasets.icon_dataset import IconDataset
from nca.data.datasets.edges_dataset import EdgesDataset
from nca.data.datasets.custom_mnist_dataset import CustomMNISTDataset
from nca.data.datasets.organizing_textures_dataset import OrganizingTexturesDataset
from nca.data.datasets.cifar10_dataset import CIFAR10Dataset
from nca.data.datasets.moving_mnist_dataset import MovingMNISTDataset
from nca.data.datasets.celeba_dataset import CelebADataset
from nca.data.datasets.growing_mnist_dataset import GrowingMNISTDataset
from nca.data.datasets.walking_dataset import WalkingDataset

__all__ = [
    "NCADataset",
    "IconDataset",
    "EdgesDataset",
    "CustomMNISTDataset",
    "OrganizingTexturesDataset",
    "CIFAR10Dataset",
    "MovingMNISTDataset",
    "CelebADataset",
    "GrowingMNISTDataset",
    "WalkingDataset",
]
