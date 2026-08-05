import torch
import argparse
import os
import pickle
import random
import numpy as np

from dfl.simulator import DFLSimulator


def main():

    parser = argparse.ArgumentParser()

    args = parser.parse_args()


    # Experiment parameters
    num_clients = 10
    rounds = 20
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


    # Fixed seed for reproducible experiments
    seed = 3

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Fixed attacker configuration
    #
    # single_attacker_id: used for the single_s1/s5/s10/s20/s50 scenarios
    # multi_attacker_ids: used for the multi_fixed scenario (strength 1.0)

    single_attacker_id = 0

    num_multi_attackers = 3

    multi_attacker_ids = random.sample(
        range(num_clients),
        num_multi_attackers
    )


    print(
        f"Single attacker id: {single_attacker_id}"
    )

    print(
        f"Multi attacker ids (strength 1.0): {multi_attacker_ids}"
    )


    simulator = DFLSimulator(
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        topology=topology,
        device=device,
        single_attacker_id=single_attacker_id,
        multi_attacker_ids=multi_attacker_ids,
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
        f"CIFAR10_"
        f"{topology}_"
        f"{num_clients}clients_"
        f"{rounds}rounds_"
        f"seed_{seed}.pkl"
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
        f"\nSaved experiment results to: {filepath}"
    )


    print("\nFinal client accuracies:")


    accuracies = simulator.evaluate()


    for i, acc in enumerate(accuracies):

        print(
            f"Client {i}: {acc:.4f}"
        )


if __name__ == "__main__":
    main()