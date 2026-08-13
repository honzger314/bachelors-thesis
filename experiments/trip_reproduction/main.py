import argparse
import os
import pickle
import random

import numpy as np
import torch

from dfl.simulator import DFLSimulator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_audit_statistics(simulator):

    for scenario_name, coordinator in simulator.coordinators.items():

        audit_log = coordinator.get_audit_log()

        total_audits = 0
        total_random = 0
        total_outlier = 0
        total_flagged = 0

        for round_log in audit_log:

            for outcome in round_log.values():

                if outcome["audited"]:

                    total_audits += 1

                    if outcome["audit_reason"] == "random":
                        total_random += 1

                    elif outcome["audit_reason"] == "outlier":
                        total_outlier += 1

                if outcome["flagged"]:
                    total_flagged += 1

        print(f"\n--- {scenario_name} ---")
        print(f"Audits:          {total_audits}")
        print(f"Random:          {total_random}")
        print(f"Outlier:         {total_outlier}")
        print(f"Flagged:         {total_flagged}")


def main():

    parser = argparse.ArgumentParser()
    parser.parse_args()

    ###########################################################
    # Fixed experiment parameters
    ###########################################################

    NUM_CLIENTS = 10
    ROUNDS = 20
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    TOPOLOGY = "ring"

    OUTLIER_THRESHOLD = 0.001

    SEEDS = [1, 2, 3]

    AUDIT_PROBABILITIES = [
        0.05,
        0.10,
        0.20,
        0.50,
        1.00,
    ]

    ###########################################################

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    total_experiments = (
        len(SEEDS)
        * len(AUDIT_PROBABILITIES)
    )

    experiment = 1

    ###########################################################
    # Run every experiment
    ###########################################################

    for seed in SEEDS:

        set_seed(seed)

        single_attacker_id = 0

        num_multi_attackers = 3

        multi_attacker_ids = random.sample(
            range(NUM_CLIENTS),
            num_multi_attackers,
        )

        for audit_probability in AUDIT_PROBABILITIES:

            print("\n" + "=" * 70)
            print(
                f"Experiment {experiment}/{total_experiments}"
            )
            print(
                f"Seed={seed} | "
                f"p={audit_probability}"
            )
            print("=" * 70)

            simulator = DFLSimulator(
                num_clients=NUM_CLIENTS,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                topology=TOPOLOGY,
                device=device,
                single_attacker_id=single_attacker_id,
                multi_attacker_ids=multi_attacker_ids,
                audit_probability=audit_probability,
                outlier_threshold=OUTLIER_THRESHOLD,
                seed=seed,
            )

            simulator.train()

            history = simulator.get_history()

            print_audit_statistics(simulator)

            print("\nAccuracy history:")
            print(history["accuracy"])

            filename = (
                f"CIFAR10_"
                f"seed{seed}_"
                f"p{audit_probability:.2f}.pkl"
            )

            filepath = os.path.join(
                "results",
                filename,
            )

            with open(filepath, "wb") as f:
                pickle.dump(history, f)

            print(
                f"\nSaved to {filepath}"
            )

            experiment += 1

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()