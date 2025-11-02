import itertools
from nca.utils.config import Config


class DataWrapper:
    def __init__(self, dataloader, config: Config):
        self.raw_dataloader = dataloader
        if config.TRAINING.STEPS == -1:
            # Infinite dataloader when epochs == -1
            dataloader = infinite_dataloader(dataloader)
        else:
            # Finite dataloader based on total samples
            dataloader = finite_dataloader(dataloader, config.TRAINING.STEPS)

        self.dataloader = dataloader
        self.config = config

    def __iter__(self):
        return self.dataloader

    def get_dataloader(self):
        return self.dataloader

    def get_dataset(self):
        return self.raw_dataloader.dataset


def infinite_dataloader(dataloader):
    """Returns an iterator that loops infinitely over the dataloader."""
    while True:
        for batch in dataloader:
            yield batch


def finite_dataloader(dataloader, total_samples):
    """Returns an iterator that loops over a dataloader for a specified number of total samples."""
    return itertools.islice(infinite_dataloader(dataloader), total_samples)