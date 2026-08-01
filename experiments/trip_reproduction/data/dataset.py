import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transform():
    """
    CIFAR-10 preprocessing.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])


def load_cifar10(
    data_path="./data"
):
    """
    Downloads and loads CIFAR-10.

    Returns:
        train_dataset
        test_dataset
    """

    transform = get_transform()

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform
    )

    return train_dataset, test_dataset



def split_dataset(
    dataset,
    num_clients,
    seed=42
):
    """
    IID split of dataset among clients.

    Every client receives an equal number
    of randomly sampled training examples.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    total_size = len(dataset)

    indices = torch.randperm(
        total_size,
        generator=generator
    ).tolist()


    split_size = total_size // num_clients


    client_datasets = []


    for i in range(num_clients):

        start = i * split_size

        if i == num_clients - 1:
            end = total_size
        else:
            end = (i + 1) * split_size


        client_indices = indices[start:end]


        client_dataset = Subset(
            dataset,
            client_indices
        )


        client_datasets.append(
            client_dataset
        )


    return client_datasets



def create_client_loaders(
    num_clients,
    batch_size=64,
    data_path="/content/drive/MyDrive/datasets",
    seed=42
):
    """
    Creates CIFAR-10 dataloaders for clients.

    Returns:

    [
        client0_loader,
        client1_loader,
        ...
    ]

    and shared test loader.
    """


    train_dataset, test_dataset = load_cifar10(
        data_path
    )


    client_datasets = split_dataset(
        train_dataset,
        num_clients,
        seed
    )


    client_loaders = []


    for dataset in client_datasets:

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        client_loaders.append(
            loader
        )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return client_loaders, test_loader