"""
MNIST data loader using torchvision.
Downloads to the per-machine cache directory resolved by config.get_data_dir().
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, get_data_dir, print_data_info


def get_mnist_loaders(batch_size: int = BATCH_SIZE):
    """
    Return (train_loader, test_loader) for MNIST.
    Data is stored in the per-machine cache, not inside the repo.
    """
    data_dir = get_data_dir()
    print_data_info(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1] float, shape (1, 28, 28)
    ])

    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader
