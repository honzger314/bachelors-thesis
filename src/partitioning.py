import numpy as np
from collections import defaultdict


def iid_split(dataset, num_clients):
    """
    Simple IID split.
    Distributes samples evenly across clients.
    """

    shards = [[] for _ in range(num_clients)]

    for i, sample in enumerate(dataset):
        shards[i % num_clients].append(sample)

    return shards


def dirichlet_split(dataset, num_clients, alpha=0.5, seed=42):
    """
    Dirichlet non-IID partitioning.

    Parameters
    ----------
    dataset : torchvision dataset
        Dataset containing (x, y) pairs.
    num_clients : int
        Number of clients.
    alpha : float
        Dirichlet concentration parameter.
        Smaller alpha => more non-IID.
    seed : int
        Random seed.

    Returns
    -------
    list
        List of client datasets.
    """

    np.random.seed(seed)

    # ------------------------
    # Extract labels
    # ------------------------
    labels = np.array([label for _, label in dataset])

    num_classes = len(np.unique(labels))

    # client_indices[i] will store dataset indices
    client_indices = [[] for _ in range(num_clients)]

    # ------------------------
    # Split class-by-class
    # ------------------------
    for cls in range(num_classes):

        cls_indices = np.where(labels == cls)[0]

        np.random.shuffle(cls_indices)

        # Sample client proportions
        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # Convert proportions into split points
        split_points = (
            np.cumsum(proportions)[:-1] * len(cls_indices)
        ).astype(int)

        splits = np.split(cls_indices, split_points)

        for client_id in range(num_clients):
            client_indices[client_id].extend(
                splits[client_id].tolist()
            )

    # ------------------------
    # Build client datasets
    # ------------------------
    client_datasets = []

    for idxs in client_indices:

        np.random.shuffle(idxs)

        client_dataset = [
            dataset[idx]
            for idx in idxs
        ]

        client_datasets.append(client_dataset)

    return client_datasets


def print_partition_stats(client_datasets):
    """
    Prints class distribution per client.
    Useful for verifying non-IIDness.
    """

    for client_id, dataset in enumerate(client_datasets):

        counts = defaultdict(int)

        for _, label in dataset:
            counts[int(label)] += 1

        print(f"\nClient {client_id}")
        print(dict(sorted(counts.items())))