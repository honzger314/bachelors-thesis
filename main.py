import torch
import copy
import random
import numpy as np

from torchvision import datasets, transforms

from client import Client
from server import Server
from model import CNN

from incentives import IncentiveTracker
from partitioning import dirichlet_split


# ------------------------
# SEEDING
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------------
# DATA
# ------------------------
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train, test


agents = 10


# ------------------------
# MAIN
# ------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train, test = load_data()

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False
    )

    # ------------------------
    # Partitioning (NON-IID)
    # ------------------------
    shards = dirichlet_split(
        train,
        num_clients=agents,
        alpha=0.5,   # FIXED (was 0 → invalid)
        seed=42
    )

    # ------------------------
    # Clients
    # ------------------------
    clients = [
        Client(i, shards[i], CNN, device=device)
        for i in range(agents)
    ]

    # ------------------------
    # Server
    # ------------------------
    server = Server(clients, device)

    global_model = CNN().to(device)

    tracker = IncentiveTracker(num_clients=agents)

    rounds = 20


    # ------------------------
    # TRAIN LOOP
    # ------------------------
    for r in range(rounds):
        print(f"\n--- Round {r} ---")

        old_weights = copy.deepcopy(global_model.state_dict())

        client_updates = []

        # ------------------------
        # Client training
        # ------------------------
        for c in clients:
            update = c.train(global_model)
            client_updates.append(update)

        # ------------------------
        # FedAvg
        # ------------------------
        new_weights, fedavg_weights = server.fedavg(client_updates)

        global_model.load_state_dict(new_weights)
        server.set_weights(new_weights)

        # ------------------------
        # LOO evaluation
        # ------------------------
        full_acc, loo_scores = server.loo_evaluate(
            client_updates,
            lambda m: server.evaluate_model(m, test_loader, device)
        )

        # ------------------------
        # Global delta
        # ------------------------
        global_delta = {
            k: new_weights[k] - old_weights[k]
            for k in old_weights
        }

        # ------------------------
        # Incentives
        # ------------------------
        round_scores = tracker.update(client_updates, global_delta)

        print("Incentive scores:", round_scores)
        print("Ranking:", tracker.ranking())

        # ------------------------
        # Final evaluation
        # ------------------------
        acc = server.evaluate(test_loader, device)

        print("Accuracy:", acc)
        print("FedAvg weights:", fedavg_weights)
        print("LOO scores:", loo_scores)


if __name__ == "__main__":
    main()