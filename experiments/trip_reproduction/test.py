import pickle
from pathlib import Path

import numpy as np


# ================================================================
# Configuration
# ================================================================

RESULTS_DIR = Path("results2")

SEEDS = [1, 2, 3]

TOPOLOGY = "ring"
NUM_CLIENTS = 10
ROUNDS = 20

AUDIT_PROBABILITY = 0.20

THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.10,
]

STEALTH_SCENARIOS = [
    "stealth_half",
    "stealth_full",
]


# ================================================================
# Result loading
# ================================================================

def result_path(seed):
    return (
        RESULTS_DIR
        / f"CIFAR10_{TOPOLOGY}_{NUM_CLIENTS}clients_"
        f"{ROUNDS}rounds_seed{seed}.pkl"
    )


def load_result(path):
    print(f"Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find result file:\n{path}"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_results():
    results = []

    for seed in SEEDS:
        results.append(
            load_result(result_path(seed))
        )

    print(f"\nLoaded {len(results)} result files.")

    return results


# ================================================================
# Helpers
# ================================================================

def scenario_key(
    base_scenario,
    audit_probability,
    threshold,
):
    threshold_string = (
        "None"
        if threshold is None
        else str(threshold)
    )

    return (
        f"{base_scenario}"
        f"__p_{audit_probability}"
        f"__threshold_{threshold_string}"
    )


# ================================================================
# Outlier detection statistics
# ================================================================

def get_stealth_outlier_detection_rate(
    history,
    scenario,
    audit_probability,
    threshold,
):
    """
    Calculate how often the stealth attacker was flagged
    specifically by the outlier detector.

    This deliberately does NOT count random audits.

    Returns:

        outlier_detection_rate =
            number of malicious reports flagged
            ---------------------------------
            total malicious reports

    Only `flagged=True` AND `outlier_audit=True`
    count as successful outlier detections.
    """

    audit_logs = history.get("audit_logs", {})

    key = scenario_key(
        scenario,
        audit_probability,
        threshold,
    )

    if key not in audit_logs:
        return np.nan

    # Identify malicious clients.
    attacker_ids = set()

    single_attacker = history.get(
        "single_attacker_id"
    )

    if single_attacker is not None:
        attacker_ids.add(single_attacker)

    attacker_ids.update(
        history.get("multi_attacker_ids", [])
    )

    if not attacker_ids:
        return np.nan

    malicious_reports = 0
    malicious_outlier_detections = 0

    for round_log in audit_logs[key]:

        if not isinstance(round_log, dict):
            continue

        for client_id, outcome in round_log.items():

            if not isinstance(outcome, dict):
                continue

            if client_id not in attacker_ids:
                continue

            malicious_reports += 1

            outlier_audit = bool(
                outcome.get(
                    "outlier_audit",
                    False,
                )
            )

            flagged = bool(
                outcome.get(
                    "flagged",
                    False,
                )
            )

            # We only count an actual outlier-based
            # successful detection.
            if outlier_audit and flagged:
                malicious_outlier_detections += 1

    if malicious_reports == 0:
        return np.nan

    return (
        malicious_outlier_detections
        / malicious_reports
    )


# ================================================================
# Main analysis
# ================================================================

def main():

    results = load_all_results()

    print("\n")
    print("=" * 75)
    print("STEALTH ATTACK OUTLIER DETECTION")
    print("=" * 75)

    print(
        "\nThis measures ONLY detections caused by the "
        "outlier detector."
    )

    print(
        f"Random audit probability p = {AUDIT_PROBABILITY}"
    )

    for scenario in STEALTH_SCENARIOS:

        print("\n" + "-" * 75)
        print(f"Scenario: {scenario}")
        print("-" * 75)

        for threshold in THRESHOLDS:

            rates = []

            for history in results:

                rate = get_stealth_outlier_detection_rate(
                    history,
                    scenario,
                    AUDIT_PROBABILITY,
                    threshold,
                )

                rates.append(rate)

            mean = np.nanmean(rates)
            std = np.nanstd(rates)

            print(
                f"threshold={threshold:<6g} | "
                f"outlier detection rate = "
                f"{mean:.3f} ± {std:.3f} "
                f"({mean * 100:.1f}%)"
            )

    print("\n" + "=" * 75)
    print("Done.")
    print("=" * 75)


if __name__ == "__main__":
    main()