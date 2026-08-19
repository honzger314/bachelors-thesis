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

    for config_key, coordinator in simulator.coordinators.items():

        audit_log = coordinator.get_audit_log()

        total_audits = 0
        total_random = 0
        total_outlier = 0
        total_flagged = 0
        total_outlier_flags = 0
        total_both = 0

        for round_log in audit_log:

            for outcome in round_log.values():

                if outcome["audited"]:
                    total_audits += 1

                if outcome.get("random_audit", False):
                    total_random += 1

                if outcome.get("outlier_audit", False):
                    total_outlier += 1

                if outcome.get("audit_reason") == "both":
                    total_both += 1

                if outcome["flagged"]:
                    total_flagged += 1

        outlier_history = coordinator.get_outlier_history()

        for round_info in outlier_history:

            client_outliers = round_info["client_outliers"]

            total_outlier_flags += sum(client_outliers.values())

        print(f"\n--- Audit statistics: {config_key} ---")
        print(f"Total audits:       {total_audits}")
        print(f"Random triggers:    {total_random}")
        print(f"Outlier triggers:   {total_outlier}")
        print(f"Both triggers:      {total_both}")
        print(f"Outlier flags:      {total_outlier_flags}")
        print(f"Flagged clients:    {total_flagged}")


# ============================================================
# Clean-scenario false-positive reporting
# ============================================================

def report_clean_false_positive_rates(simulator):
    """
    Isolates the "clean" scenario specifically - since it has zero
    actual malicious clients by construction, any outlier flag or
    any "flagged" audit outcome there is unambiguously a false
    positive. This is the number that matters for judging whether
    an outlier_threshold is well-calibrated (as opposed to the
    outlier_flag_rate in summary_metrics, which conflates honest
    and malicious flags for attacked scenarios).
    """

    print("\n" + "=" * 70)
    print("CLEAN-SCENARIO FALSE-POSITIVE RATES")
    print("(zero actual attackers - any flag here is a false positive)")
    print("=" * 70)

    for key, coordinator in simulator.coordinators.items():

        if not key.startswith("clean__"):
            continue

        audit_log = coordinator.get_audit_log()
        outlier_history = coordinator.get_outlier_history()

        num_opportunities = (
            len(audit_log) * simulator.num_clients
            if audit_log else 0
        )

        if num_opportunities == 0:
            continue

        total_outlier_flags = sum(
            sum(round_info["client_outliers"].values())
            for round_info in outlier_history
        )

        total_flagged = sum(
            1
            for round_log in audit_log
            for outcome in round_log.values()
            if outcome["flagged"]
        )

        outlier_fp_rate = total_outlier_flags / num_opportunities
        flagged_fp_rate = total_flagged / num_opportunities

        print(f"\n{key}")
        print(
            f"  Outlier-heuristic false-positive rate: "
            f"{outlier_fp_rate:.2%} "
            f"({total_outlier_flags}/{num_opportunities})"
        )
        print(
            f"  Final flagged-malicious false-positive rate: "
            f"{flagged_fp_rate:.2%} "
            f"({total_flagged}/{num_opportunities})"
        )

        if total_flagged > 0:
            print(
                "  WARNING: clean scenario produced flagged=True "
                "at least once - this should only be possible if "
                "audit_threshold is too tight for genuine "
                "floating-point noise, or indicates a bug. "
                "Investigate before trusting downstream results."
            )


# ============================================================
# Theoretical threshold reporting
# ============================================================

def report_theoretical_thresholds(simulator):

    thresholds = simulator.history.get("theoretical_thresholds")

    if not thresholds:
        return

    e_estimate = simulator.history.get("honest_reward_estimate")
    n_neighbors = simulator.history.get("n_neighbors_avg")
    n_total = simulator.history.get("n_total")

    print("\n" + "=" * 70)
    print("THEORETICAL THRESHOLD REFERENCE VALUES")
    print("(t <= p*e / (n*(1-p)) - reference only, NOT enforced)")
    print("=" * 70)

    print(f"Estimated honest reward e: {e_estimate:.6f}")
    print(f"n (with neighbors, avg):   {n_neighbors:.2f}")
    print(f"n (without neighbors):     {n_total}")
    print(
        f"Actual audit_threshold used (numerical tolerance): "
        f"{simulator.audit_threshold:.2e}"
    )

    for p, variants in thresholds.items():

        with_n = variants["with_neighbors"]
        without_n = variants["without_neighbors"]

        with_n_str = (
            f"{with_n:.6f}" if with_n is not None else "vacuous (p=1)"
        )
        without_n_str = (
            f"{without_n:.6f}"
            if without_n is not None else "vacuous (p=1)"
        )

        print(
            f"  p={p:<5} "
            f"t_with_neighbors={with_n_str:<20} "
            f"t_without_neighbors={without_n_str}"
        )


# ============================================================
# Main experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.parse_args()

    # ========================================================
    # Fixed experiment parameters
    # ========================================================

    NUM_CLIENTS = 10
    ROUNDS = 20
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    TOPOLOGY = "ring"

    SEEDS = [1, 2, 3]

    # --------------------------------------------------------
    # Probabilistic audit probabilities
    #
    # p=0.0 included so the grid contains, for every threshold:
    #   - full honest baseline (clean, p=0, threshold=None)
    #   - outlier detection ONLY (p=0, threshold set)
    #   - random auditing ONLY (p>0, threshold=None)
    #   - combined defense (p>0, threshold set)
    # --------------------------------------------------------

    AUDIT_PROBABILITIES = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.50,
        1.00,
    ]

    # --------------------------------------------------------
    # Outlier thresholds. None disables the heuristic completely.
    # --------------------------------------------------------

    OUTLIER_THRESHOLDS = [
        None,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    # --------------------------------------------------------
    # Numerical-precision tolerance for the audit accept/reject
    # test. NOT the economic threshold - see Coordinator's
    # docstring. Kept explicit and named rather than hardcoded.
    # --------------------------------------------------------

    AUDIT_TOLERANCE = 1e-6

    # ========================================================
    # Device
    # ========================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ========================================================
    # Result directory
    # ========================================================

    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 70)
    print("FULL EXPERIMENT GRID")
    print("=" * 70)
    print(f"Seeds:               {SEEDS}")
    print(f"Audit probabilities: {AUDIT_PROBABILITIES}")
    print(f"Outlier thresholds:  {OUTLIER_THRESHOLDS}")
    print(f"Audit tolerance:     {AUDIT_TOLERANCE}")
    print(f"Rounds per run:      {ROUNDS}")
    print(f"Simulator (training) runs: {len(SEEDS)}")
    print("=" * 70)

    for seed in SEEDS:

        set_seed(seed)

        single_attacker_id = 0
        num_multi_attackers = 3

        multi_attacker_ids = random.sample(
            range(NUM_CLIENTS),
            num_multi_attackers,
        )

        print("\n" + "#" * 70)
        print(f"SEED {seed}")
        print(f"Single attacker: {single_attacker_id}")
        print(f"Multi attackers: {multi_attacker_ids}")
        print("#" * 70)

        simulator = DFLSimulator(
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            topology=TOPOLOGY,
            device=device,
            single_attacker_id=single_attacker_id,
            multi_attacker_ids=multi_attacker_ids,
            audit_probabilities=AUDIT_PROBABILITIES,
            outlier_thresholds=OUTLIER_THRESHOLDS,
            audit_threshold=AUDIT_TOLERANCE,
            seed=seed,
        )

        print(
            f"\nActive coordinator configurations: "
            f"{len(simulator.coordinators)} "
            f"(skipped {len(simulator.skipped_configs)} "
            f"stealth/threshold=None combinations)"
        )

        simulator.train()

        # ----------------------------------------------------
        # Post-training analysis: theoretical thresholds and
        # summary metrics, both saved into history.
        # ----------------------------------------------------

        simulator.compute_theoretical_thresholds()
        simulator.compute_summary_metrics()

        history = simulator.get_history()

        print_audit_statistics(simulator)
        report_clean_false_positive_rates(simulator)
        report_theoretical_thresholds(simulator)

        print("\nAccuracy history:")
        print(history["accuracy"])

        filename = (
            f"CIFAR10_{TOPOLOGY}_{NUM_CLIENTS}clients_"
            f"{ROUNDS}rounds_seed{seed}.pkl"
        )

        filepath = os.path.join("results", filename)

        with open(filepath, "wb") as f:
            pickle.dump(history, f)

        print(f"\nSaved to:\n{filepath}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()