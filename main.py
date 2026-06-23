import torch
import copy
import random
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import random_split

from client import Client
from server import Server
from model import CNN

from incentives import IncentiveTracker
from partitioning import dirichlet_split


# ------------------------
# SEED
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# DATA
# ------------------------
def load_data(val_ratio=0.1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    # ------------------------
    # split train → train + val
    # ------------------------
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train, val = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train, val, test

# ------------------------
# CORE EXPERIMENT FUNCTION
# ------------------------
def run_experiment(
    agents=10,
    alpha=0.5,
    rounds=20,
    seed=42,
    device=None,
    return_history=True
):

    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    train, val, test = load_data()

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=64,
        shuffle=False
    )

    # ------------------------
    # Partition
    # ------------------------
    shards = dirichlet_split(
        train,
        num_clients=agents,
        alpha=alpha,
        seed=seed
    )

    # ------------------------
    # Setup
    # ------------------------
    clients = [
        Client(i, shards[i], CNN, device=device)
        for i in range(agents)
    ]

    server = Server(clients, device)

    global_model = CNN().to(device)

    tracker = IncentiveTracker(num_clients=agents)

    history = []

    # ------------------------
    # TRAIN LOOP
    # ------------------------
    for r in range(rounds):
        print(f"\n--- Round {r} ---")

        old_weights = copy.deepcopy(global_model.state_dict())

        client_updates = []

        for c in clients:
            update = c.train(global_model.to(device))
            client_updates.append(update)

        # FedAvg
        new_weights, fedavg_weights = server.fedavg(client_updates)

        global_model.load_state_dict(new_weights)
        server.set_weights(new_weights)

        # LOO
        full_acc, loo_scores = server.loo_evaluate(
            client_updates,
            lambda m: server.evaluate_model(m, val_loader, device)
        )

        global_delta = {
            k: new_weights[k] - old_weights[k]
            for k in old_weights
        }

        # Incentives
        round_scores = tracker.update(client_updates, global_delta)

        acc = server.evaluate(test_loader, device)

        print("Accuracy:", acc)

        record = {
            "round": r,
            "accuracy": acc,
            "incentives": round_scores,
            "ranking": tracker.ranking(),
            "loo": loo_scores
        }

        history.append(record)

    # ------------------------
    # OUTPUT
    # ------------------------
    result = {
        "final_accuracy": history[-1]["accuracy"],
        "final_ranking": history[-1]["ranking"],
        "history": history
    }

    if return_history:
        return result, history

    return result


# optional local run
if __name__ == "__main__":
    run_experiment()