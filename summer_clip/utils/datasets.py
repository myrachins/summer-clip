import typing as tp

from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision.datasets import ImageNet


class NoImageImageNetDataset(ImageNet):
    def __init__(self, *args, **kwargs) -> None:
        kwargs = {**kwargs, **{'loader': lambda _: None}}
        super().__init__(*args, **kwargs)


class ImageNetV2Wrapper(ImageNetV2Dataset):
    def __init__(self, *args, image_net_root: tp.Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classes = None

        if image_net_root is not None:
            image_net = ImageNet(image_net_root, split='val')
            self.classes = image_net.classes
