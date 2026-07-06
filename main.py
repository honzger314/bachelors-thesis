import torch
import random
import numpy as np
import copy

from torchvision import datasets, transforms
from torch.utils.data import random_split

from client import Client
from server import Server
from model import CNN
from partitioning import dirichlet_split
from telemetry import TelemetryCollector

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
def load_data(seed, val_ratio=0.1):
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

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train, val = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train, val, test


# ------------------------
# CORE EXPERIMENT
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

    train, val, test = load_data(seed)

    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False)

    shards = dirichlet_split(
        train,
        num_clients=agents,
        alpha=alpha,
        seed=seed
    )

    clients = [
        Client(i, shards[i], CNN, device=device)
        for i in range(agents)
    ]

    server = Server(clients, device)
    global_model = CNN().to(device)

    telemetry = TelemetryCollector()

    history = []

    # ------------------------
    # TRAIN LOOP
    # ------------------------
    for r in range(rounds):
        print(f"\n--- Round {r} ---")

        old_weights = copy.deepcopy(global_model.state_dict())

        client_updates = []

        for c in clients:
            state, update, grad = c.train(global_model)
            client_updates.append((state, update, grad))

        # split tuples
        client_states = [c[0] for c in client_updates]
        client_grads = [c[2] for c in client_updates]

        # FedAvg
        new_weights, _ = server.fedavg(client_updates)

        global_model.load_state_dict(new_weights)
        server.set_weights(new_weights)

        # LOO
        full_acc, loo_scores = server.loo_evaluate(
            client_updates,
            lambda m: server.evaluate_model(m, val_loader, device)
        )

        acc = server.evaluate(test_loader, device)

        # ------------------------
        # TELEMETRY LOGGING
        # ------------------------
        telemetry.log_round(
            round_id=r,
            global_model=global_model,
            client_updates=client_states,
            client_grads=client_grads,
            client_losses_before=None,
            client_losses_after=None,
            test_loss=None,
            test_acc=acc,
            loo_scores=loo_scores
        )

        print("Accuracy:", acc)

        history.append({
            "round": r,
            "accuracy": acc,
            "loo": loo_scores
        })

    df = telemetry.export()

    result = {
        "final_accuracy": history[-1]["accuracy"],
        "history": history
    }

    if return_history:
        return result, df  # return DataFrame separately

    return result, df


# optional run
if __name__ == "__main__":
    run_experiment()