import torch
import torch.nn as nn
import itertools
from .models_registry import generators


class Generator(nn.Module):
    def __init__(self, z_dim, M):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 1024, M, 1, 0, bias=False),  # 4, 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(-1, self.z_dim, 1, 1))

    def generate(self, n, batch_size, loader_z):
        num_batches = n // batch_size + 1
        with torch.no_grad():
            gen_imgs = [
                self(z) for z in itertools.islice(loader_z, num_batches)
            ]
            gen_imgs = torch.cat(gen_imgs)
            return gen_imgs[:n]



@generators.add_to_registry("dcgan")
class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=2)