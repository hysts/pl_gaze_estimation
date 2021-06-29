from typing import Callable

import torchvision.transforms as T


def create_transform() -> Callable:
    transforms = [
        T.ToTensor(),
    ]
    return T.Compose(transforms)
