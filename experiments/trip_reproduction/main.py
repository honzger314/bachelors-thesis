import argparse
import os
import pickle
import random

import numpy as np
import torch

from dfl.simulator import DFLSimulator


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Audit statistics
# ============================================================

def print_audit_statistics(simulator):

    for config_key, coordinator in (
        simulator.coordinators.items()
    ):

        audit_log = coordinator.get_audit_log()

        total_audits = 0

        # Random component of audits.
        #
        # This counts random_audit=True, including cases where
        # the same client was also flagged by the heuristic.
        #
        total_random = 0

        # Outlier component of audits.
        #
        # This counts outlier_audit=True, including cases where
        # the random audit also triggered.
        #
        total_outlier = 0

        # Number of dishonest reports actually caught.
        total_flagged = 0

        # Honest reports incorrectly flagged by the heuristic.
        total_outlier_flags = 0

        # Audits triggered by both mechanisms simultaneously.
        total_both = 0

        for round_log in audit_log:

            for outcome in round_log.values():

                if outcome["audited"]:

                    total_audits += 1

                if outcome.get(
                    "random_audit",
                    False
                ):
                    total_random += 1

                if outcome.get(
                    "outlier_audit",
                    False
                ):
                    total_outlier += 1

                if outcome.get(
                    "audit_reason"
                ) == "both":
                    total_both += 1

                if outcome["flagged"]:
                    total_flagged += 1

                    # A flagged client was dishonest, so this
                    # is NOT a false positive.
                    #
                    # The heuristic itself can still be useful
                    # to analyze separately through the outlier
                    # history below.

        # ----------------------------------------------------
        # Count heuristic outlier detections directly from the
        # coordinator history.
        # ----------------------------------------------------

        outlier_history = (
            coordinator.get_outlier_history()
        )

        for round_info in outlier_history:

            client_outliers = (
                round_info["client_outliers"]
            )

            total_outlier_flags += sum(
                client_outliers.values()
            )

        print(
            f"\n--- Audit statistics: "
            f"{config_key} ---"
        )

        print(
            f"Total audits:       {total_audits}"
        )

        print(
            f"Random triggers:    {total_random}"
        )

        print(
            f"Outlier triggers:   {total_outlier}"
        )

        print(
            f"Both triggers:      {total_both}"
        )

        print(
            f"Outlier flags:      {total_outlier_flags}"
        )

        print(
            f"Flagged clients:    {total_flagged}"
        )


# ============================================================
# Main experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # There are currently no command-line arguments, but keeping
    # argparse makes it easy to add them later.
    parser.parse_args()

    # ========================================================
    # Fixed experiment parameters
    # ========================================================

    NUM_CLIENTS = 10

    ROUNDS = 20

    LOCAL_EPOCHS = 1

    BATCH_SIZE = 64

    TOPOLOGY = "ring"

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    SEEDS = [
        1,
        2,
        3,
    ]

    # --------------------------------------------------------
    # Probabilistic audit probabilities
    # --------------------------------------------------------

    AUDIT_PROBABILITIES = [
        0.05,
        0.10,
        0.20,
        0.50,
        1.00,
    ]

    # --------------------------------------------------------
    # Outlier thresholds
    #
    # None disables the heuristic completely.
    #
    # 0.001 is the previous threshold which caused very high
    # false-positive/outlier rates (>90% of honest reports).
    #
    # The sweep is shifted upward from that point to find where
    # the heuristic actually becomes selective.
    # --------------------------------------------------------

    OUTLIER_THRESHOLDS = [
        None,
        0.001,
        0.005,
        0.01,
        0.05,
        0.10,
    ]

    # ========================================================
    # Device
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Result directory
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    # ========================================================
    # Experiment count
    #
    # Each simulator run trains ONCE per seed and internally
    # evaluates every combination of:
    #
    #   attack scenario (10)
    #       × audit probability (5)
    #       × outlier threshold (6)
    #   = 300 coordinator configurations per seed
    #
    # Training/LCV computation is NOT repeated across those
    # 300 configurations -- only 3 simulator.train() calls
    # happen in total (one per seed).
    # ========================================================

    num_configs_per_seed = (
        len(AUDIT_PROBABILITIES)
        * len(OUTLIER_THRESHOLDS)
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL EXPERIMENT GRID"
    )

    print(
        "=" * 70
    )

    print(
        f"Seeds:              {SEEDS}"
    )

    print(
        f"Audit probabilities: {AUDIT_PROBABILITIES}"
    )

    print(
        f"Outlier thresholds:  {OUTLIER_THRESHOLDS}"
    )

    print(
        f"Rounds per run:      {ROUNDS}"
    )

    print(
        f"Simulator (training) runs: "
        f"{len(SEEDS)}"
    )

    print(
        f"Coordinator configs per seed: "
        f"{num_configs_per_seed} "
        f"(× attack scenarios, evaluated internally)"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Experiment grid
    #
    # Training happens exactly once per seed. The audit
    # probability and outlier threshold sweeps are passed in
    # as lists and consumed INSIDE the simulator, which reuses
    # the same trained models / ground-truth LCVs for every
    # combination.
    # ========================================================

    for seed in SEEDS:

        # ----------------------------------------------------
        # Set all random seeds.
        # ----------------------------------------------------

        set_seed(seed)

        # ----------------------------------------------------
        # Fixed attacker configuration for this seed.
        # ----------------------------------------------------

        single_attacker_id = 0

        num_multi_attackers = 3

        multi_attacker_ids = random.sample(
            range(NUM_CLIENTS),
            num_multi_attackers,
        )

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"SEED {seed}"
        )

        print(
            f"Single attacker: {single_attacker_id}"
        )

        print(
            f"Multi attackers: {multi_attacker_ids}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------
        # Build ONE simulator for this seed, covering the
        # entire p / threshold sweep. simulator.train() below
        # performs local training and LCV computation exactly
        # once per round, then fans that result out across
        # every (scenario, p, threshold) coordinator.
        # ----------------------------------------------------

        simulator = DFLSimulator(
            num_clients=NUM_CLIENTS,

            rounds=ROUNDS,

            local_epochs=LOCAL_EPOCHS,

            batch_size=BATCH_SIZE,

            topology=TOPOLOGY,

            device=device,

            single_attacker_id=(
                single_attacker_id
            ),

            multi_attacker_ids=(
                multi_attacker_ids
            ),

            audit_probabilities=(
                AUDIT_PROBABILITIES
            ),

            outlier_thresholds=(
                OUTLIER_THRESHOLDS
            ),

            seed=seed,
        )

        # ----------------------------------------------------
        # Run training ONCE for this seed.
        # ----------------------------------------------------

        simulator.train()

        # ----------------------------------------------------
        # Retrieve experiment history (contains every
        # scenario/p/threshold combination).
        # ----------------------------------------------------

        history = (
            simulator.get_history()
        )

        # ----------------------------------------------------
        # Print audit statistics for every configuration.
        # ----------------------------------------------------

        print_audit_statistics(
            simulator
        )

        # ----------------------------------------------------
        # Print accuracy trajectory.
        #
        # There is only one accuracy trajectory per seed,
        # since the contribution-defense configuration does
        # not affect model aggregation.
        # ----------------------------------------------------

        print(
            "\nAccuracy history:"
        )

        print(
            history["accuracy"]
        )

        # ========================================================
        # Save ONE result file per seed, containing every
        # scenario / p / threshold combination.
        # ========================================================

        filename = (
            f"CIFAR10_"
            f"{TOPOLOGY}_"
            f"{NUM_CLIENTS}clients_"
            f"{ROUNDS}rounds_"
            f"seed{seed}.pkl"
        )

        filepath = os.path.join(
            "results",
            filename,
        )

        with open(
            filepath,
            "wb"
        ) as f:

            pickle.dump(
                history,
                f
            )

        print(
            f"\nSaved to:"
            f"\n{filepath}"
        )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL EXPERIMENTS FINISHED"
    )

    print(
        f"Completed "
        f"{len(SEEDS)} "
        f"simulator (training) runs, each covering "
        f"{num_configs_per_seed} p/threshold combinations "
        f"× {len(simulator.scenarios)} attack scenarios."
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()