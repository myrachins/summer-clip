import torch
import torch.utils.data
import torch.distributions
import torchvision
import utils

priors = utils.ClassRegistry()
datasets = utils.ClassRegistry()
loaders = utils.ClassRegistry()


@datasets.add_to_registry("cifar10", ("train", "test"))
class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        super().__init__(root, transform=transform, train=train)


@priors.add_to_registry("normal")
class PriorDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dim):
        self.dist = torch.distributions.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        )

    def __getitem__(self, index):
        return self.dist.sample()

    def __len__(self):
        return int(1e6)


@loaders.add_to_registry("infinite", ("prior", "train", "test"))
class InfiniteLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=True,
        device=None,
        pin_memory=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            multiprocessing_context="fork" if num_workers > 0 else None,
            **kwargs,
        )
        self.infinite = infinite
        self.device = device
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            if self.infinite:
                self.dataset_iterator = super().__iter__()
                x = next(self.dataset_iterator)
            else:
                raise
        if self.device is not None:
            x = utils.move_to_device(x, self.device)
        return x
