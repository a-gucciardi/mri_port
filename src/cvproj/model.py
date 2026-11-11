from torch import Tensor, nn


class Autoencoder(nn.Module):
    def __init__(self, width: int, channels: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)
