import itertools
import torch
import torch.nn as nn
from .models_registry import generators


_N_CHANNELS = 3
_GEN_SIZE = 256


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


@generators.add_to_registry("resnet_v2")
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * _GEN_SIZE)
        self.final = nn.Conv2d(_GEN_SIZE, _N_CHANNELS, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(_GEN_SIZE, _GEN_SIZE, stride=2),
            ResBlockGenerator(_GEN_SIZE, _GEN_SIZE, stride=2),
            ResBlockGenerator(_GEN_SIZE, _GEN_SIZE, stride=2),
            nn.BatchNorm2d(_GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, _GEN_SIZE, 4, 4))

    def generate(self, n, batch_size, loader_z):
        num_batches = n // batch_size + 1
        with torch.no_grad():
            gen_imgs = [
                self(z) for z in itertools.islice(loader_z, num_batches)
            ]
            gen_imgs = torch.cat(gen_imgs)
            return gen_imgs[:n]
