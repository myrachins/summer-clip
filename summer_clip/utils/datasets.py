import typing as tp

from torch.utils.data.dataset import Dataset
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision import transforms
from torchvision.datasets import ImageNet

from summer_clip.tip_adapter.datasets import build_dataset
from summer_clip.tip_adapter.datasets.utils import DatasetBase
from summer_clip.tip_adapter.datasets.utils import DatasetWrapper


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


class TipAdapterDataset(Dataset):
    def __init__(self, dataset: str, split: str, root_path: str, shots: int,
                 input_size: int = 224, is_train: bool = False, use_custom_preprocess: bool = False,
                 load_images: bool = True, transform: tp.Optional[tp.Any] = None) -> None:
        super().__init__()
        self.tip_base_dataset = build_dataset(dataset, root_path, shots)
        split_data = self._select_split(self.tip_base_dataset, split)
        dataset_transform = transform if not use_custom_preprocess else self._get_custom_preprocess()
        self.dataset = DatasetWrapper(
            split_data, input_size, transform=dataset_transform,
            is_train=is_train, load_images=load_images
        )

    @staticmethod
    def _select_split(dataset: DatasetBase, split: str) -> tp.Any:
        if split == 'train':
            return dataset.train_x
        if split == 'val':
            return dataset.val
        if split == 'test':
            return dataset.test
        raise ValueError(f"Unsupported split name: '{split}'")

    @staticmethod
    def _get_custom_preprocess() -> tp.Any:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        return train_transform

    def __getitem__(self, index) -> tp.Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def classes(self) -> tp.List[str]:
        return self.tip_base_dataset.classnames
