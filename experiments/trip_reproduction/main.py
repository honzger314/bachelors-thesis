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
    rounds = 5
    local_epochs = 1
    batch_size = 64
    topology = "ring"

    # Probabilistic LCV verification
    audit_probability = 0.2
    audit_threshold = 0.001

    # Automatically use GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Fixed seed for reproducible experiments
    seed = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fixed attacker configuration
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
        f"Multi attacker ids: {multi_attacker_ids}"
    )

    print(
        f"Audit probability: {audit_probability}"
    )

    print(
        f"Audit threshold: {audit_threshold}"
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
        audit_probability=audit_probability,
        audit_threshold=audit_threshold,
        seed=seed,
    )

    clients = simulator.train()

    history = simulator.get_history()

    for scenario_name, coordinator in simulator.coordinators.items():

        audit_log = coordinator.get_audit_log()

        total_audits = 0
        total_random_audits = 0
        total_outlier_audits = 0
        total_flagged = 0

        for round_log in audit_log:
            for outcome in round_log.values():

                if outcome["audited"]:
                    total_audits += 1

                    if outcome["audit_reason"] == "random":
                        total_random_audits += 1

                    elif outcome["audit_reason"] == "outlier":
                        total_outlier_audits += 1

                if outcome["flagged"]:
                    total_flagged += 1

        print(f"\n--- Audit statistics: {scenario_name} ---")
        print(f"Total audits:          {total_audits}")
        print(f"Random audits:         {total_random_audits}")
        print(f"Outlier audits:        {total_outlier_audits}")
        print(f"Flagged clients:       {total_flagged}")

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
        f"seed_{seed}_"
        f"p_{audit_probability}_"
        f"t_{audit_threshold}.pkl"
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