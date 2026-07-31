import torch
import argparse
import os
import pickle
import random
import numpy as np

from dfl.simulator import DFLSimulator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lcv",
        type=str,
        default="original",
        choices=[
            "original",
            "modified"
        ],
        help="Which LCV implementation to use"
    )

    args = parser.parse_args()


    # Experiment parameters
    num_clients = 10
    rounds = 10
    local_epochs = 1
    batch_size = 64
    topology = "ring"


    # Automatically use GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )


    print(
        f"Using LCV method: {args.lcv}"
    )

    # Fixed seed for reproducible experiments
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    simulator = DFLSimulator(
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        topology=topology,
        device=device,
        lcv_method=args.lcv
    )


    clients = simulator.train()


    history = simulator.get_history()


    print("\nAccuracy history:")
    print(history["accuracy"])


    #
    # Save experiment results
    #

    os.makedirs(
        "results",
        exist_ok=True
    )


    filename = (
        f"{topology}_"
        f"{num_clients}clients_"
        f"{rounds}rounds_"
        f"{args.lcv}_lcv_"
        f"seed{seed}.pkl"
    )


    filepath = os.path.join(
        "results",
        filename
    )


    with open(filepath, "wb") as f:

        pickle.dump(
            history,
            f
        )


    print(
        f"\nSaved results to: {filepath}"
    )


    print("\nFinal client accuracies:")


    accuracies = simulator.evaluate()


    for i, acc in enumerate(accuracies):

        print(
            f"Client {i}: {acc:.4f}"
        )


if __name__ == "__main__":
    main()