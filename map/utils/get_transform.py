import os

from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize

def get_transform():
    _MEAN = [0.5, 0.5, 0.5]
    _STD = [0.5, 0.5, 0.5]

    transform = Compose(
        [
            ToTensor(),
            Normalize(_MEAN, _STD),
        ]
    )

    return transform, _MEAN, _STD