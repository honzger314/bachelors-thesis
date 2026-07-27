import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transform():
    """
    Image preprocessing.

    MNIST is already normalized, but converting to tensor
    is required for PyTorch models.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])



def load_mnist(
    data_path="./data"
):
    """
    Downloads and loads MNIST.

    Returns:
        train_dataset
        test_dataset
    """

    transform = get_transform()

    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset



def split_dataset(
    dataset,
    num_clients,
    seed=42
):
    """
    Splits dataset into independent client datasets.

    Currently:
        IID split

    Example:

        60000 samples
        5 clients

        Client 0:
            12000 samples

        Client 1:
            12000 samples

        ...

    Later this function can be replaced with:
        - label skew
        - quantity skew
        - noisy labels
        - noisy images
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

        # Last client gets remaining samples
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
    data_path="./data",
    seed=42
):
    """
    Creates DataLoaders for every client.

    Returns:

    [
        loader_client_0,
        loader_client_1,
        ...
    ]
    """

    train_dataset, test_dataset = load_mnist(
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