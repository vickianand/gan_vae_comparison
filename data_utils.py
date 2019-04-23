from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_dataloader(data="svhn_train", data_dir="data/svhn/", batch_size=32):
    """
    Args:
        data (str): one of {svhn_train, svhn_test, }
        data_dir (str): path where data file should be looked for and if
            necessary should be downloaded
    """
    svhn_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    if data == "svhn_train":
        dataset = SVHN(
            root=data_dir, split="train", transform=svhn_transform, download=True
        )
    elif data == "svhn_test":
        dataset = SVHN(
            root=data_dir, split="test", transform=svhn_transform, download=True
        )
    elif data == "svhn_extra":
        dataset = SVHN(
            root=data_dir, split="extra", transform=svhn_transform, download=True
        )
    else:
        raise (NotImplementedError)

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True
    )
