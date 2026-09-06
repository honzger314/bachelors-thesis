import pickle
from pathlib import Path

import os
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Configuration
# ================================================================

RESULTS_DIR = Path("results4")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [1, 2, 3]

TOPOLOGY = "ring"
NUM_CLIENTS = 10
ROUNDS = 20

# Representative operating point used for the combined-defense
# comparison (Figure 5) and wherever a single finite (p, threshold)
# pair is needed alongside a specific isolated axis.
MAIN_AUDIT_PROBABILITY = 0.20
MAIN_THRESHOLD = 0.01

AUDIT_PROBABILITIES = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
ATTACK_STRENGTHS = [1, 5, 10, 20, 50]

STEALTH_SCENARIOS = [
    ("stealth_half", "Stealth half"),
    ("stealth_full", "Stealth full"),
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
    results = [load_result(result_path(seed)) for seed in SEEDS]
    print(f"\nLoaded {len(results)} result files.")
    return results


# ================================================================
# Scenario helpers
# ================================================================

def scenario_key(base_scenario, audit_probability, threshold):

    threshold_string = (
        "None" if threshold is None else str(threshold)
    )

    return (
        f"{base_scenario}"
        f"__p_{audit_probability}"
        f"__threshold_{threshold_string}"
    )


# ================================================================
# Contribution helpers
# ================================================================

def get_contribution_entry(
    history, base_scenario, audit_probability, threshold,
    round_idx, version,
):

    scenario = scenario_key(
        base_scenario, audit_probability, threshold
    )

    contributions = history["contributions"][scenario]
    entry = contributions[round_idx][version]

    client_ids = sorted(entry.keys())

    matrix = np.stack(
        [np.asarray(entry[cid], dtype=float) for cid in client_ids]
    )

    return matrix, client_ids


def get_final_contribution(
    history, base_scenario, audit_probability, threshold,
    version, client_id,
):
    """
    Returns one client's self-contribution (diagonal entry) at the
    final round.
    """

    matrix, client_ids = get_contribution_entry(
        history, base_scenario, audit_probability, threshold,
        ROUNDS - 1, version,
    )

    idx = client_ids.index(client_id)

    return float(matrix[idx, idx])


def get_final_aggregated_contribution(
    history, base_scenario, audit_probability, threshold,
    version, client_ids,
):
    """
    Sum of self-contributions across multiple clients at the final
    round - used for multi-attacker aggregation (Figure 1, panel 2).
    """

    return sum(
        get_final_contribution(
            history, base_scenario, audit_probability, threshold,
            version, cid,
        )
        for cid in client_ids
    )


# ================================================================
# Audit statistics
# ================================================================

def get_audit_statistics(
    history, base_scenario, audit_probability, threshold,
):
    """
    Per-experiment audit statistics, split by whether the reporting
    client was actually malicious in that scenario.
    """

    audit_logs = history.get("audit_logs", {})
    key = scenario_key(base_scenario, audit_probability, threshold)

    empty = {
        "audit_rate": np.nan,
        "malicious_audit_rate": np.nan,
        "malicious_detection_rate": np.nan,
        "honest_audit_rate": np.nan,
        "false_positive_rate": np.nan,
        "random_audit_rate": np.nan,
        "outlier_audit_rate": np.nan,
        "malicious_outlier_audit_rate": np.nan,
        "honest_outlier_audit_rate": np.nan,
    }

    if key not in audit_logs:
        return empty

    attacker_ids = set()

    single_attacker = history.get("single_attacker_id")
    if single_attacker is not None:
        attacker_ids.add(single_attacker)

    attacker_ids.update(history.get("multi_attacker_ids", []))

    total_reports = 0
    total_audited = 0
    total_random_audits = 0
    total_outlier_audits = 0

    malicious_reports = 0
    malicious_audited = 0
    malicious_detected = 0
    malicious_outlier_audits = 0

    honest_reports = 0
    honest_audited = 0
    honest_flagged = 0
    honest_outlier_audits = 0

    for round_log in audit_logs[key]:

        if not isinstance(round_log, dict):
            continue

        for client_id, outcome in round_log.items():

            if not isinstance(outcome, dict):
                continue

            total_reports += 1

            audited = bool(outcome.get("audited", False))
            random_audit = bool(outcome.get("random_audit", False))
            outlier_audit = bool(outcome.get("outlier_audit", False))
            flagged = bool(outcome.get("flagged", False))

            if audited:
                total_audited += 1
            if random_audit:
                total_random_audits += 1
            if outlier_audit:
                total_outlier_audits += 1

            if client_id in attacker_ids:

                malicious_reports += 1

                if audited:
                    malicious_audited += 1
                if flagged:
                    malicious_detected += 1
                if outlier_audit:
                    malicious_outlier_audits += 1

            else:

                honest_reports += 1

                if audited:
                    honest_audited += 1
                if flagged:
                    honest_flagged += 1
                if outlier_audit:
                    honest_outlier_audits += 1

    def safe_div(a, b):
        return a / b if b > 0 else np.nan

    return {
        "audit_rate": safe_div(total_audited, total_reports),
        "malicious_audit_rate": safe_div(
            malicious_audited, malicious_reports
        ),
        "malicious_detection_rate": safe_div(
            malicious_detected, malicious_reports
        ),
        "honest_audit_rate": safe_div(honest_audited, honest_reports),
        "false_positive_rate": safe_div(
            honest_flagged, honest_reports
        ),
        "random_audit_rate": safe_div(
            total_random_audits, total_reports
        ),
        "outlier_audit_rate": safe_div(
            total_outlier_audits, total_reports
        ),
        "malicious_outlier_audit_rate": safe_div(
            malicious_outlier_audits, malicious_reports
        ),
        "honest_outlier_audit_rate": safe_div(
            honest_outlier_audits, honest_reports
        ),
    }


# ================================================================
# Generic helpers
# ================================================================

def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values))

def print_series_table(title, x_values, means, stds, x_name):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{x_name:<25} {'Mean':>15} {'Std':>15} {'Mean ± Std':>25}")
    print("-" * 80)

    for x, mean, std in zip(x_values, means, stds):
        print(
            f"{str(x):<25} "
            f"{mean:>15.6f} "
            f"{std:>15.6f} "
            f"{mean:.6f} ± {std:.6f}"
        )

def save_figure(fig, filename):
    pdf_path = FIGURES_DIR / f"{filename}.pdf"
    png_path = FIGURES_DIR / f"{filename}.png"

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


# ================================================================
# Helper: write PGFPlots data
# ================================================================

def write_dat(filename, header, rows):
    """
    Write numerical/categorical experiment results in a format
    directly usable by pgfplots.

    Example:
        write_dat(
            "figures/data/example.dat",
            ["x", "mean", "std"],
            [(1, 0.5, 0.02), (2, 0.7, 0.03)]
        )
    """
    os.makedirs("figures/data", exist_ok=True)

    path = os.path.join("figures/data", filename)

    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")

        for row in rows:
            formatted = []

            for value in row:
                if isinstance(value, str):
                    formatted.append(value)
                else:
                    formatted.append(f"{value:.8f}")

            f.write(" ".join(formatted) + "\n")

    print(f"Saved data: {path}")


# ================================================================
# Figure 1 (RQ1) - Attack works without any defense
# ================================================================

def plot_attack_without_defense(results):

    attacker_id = results[0]["single_attacker_id"]

    means = []
    stds = []

    for strength in ATTACK_STRENGTHS:

        values = [
            get_final_contribution(
                history,
                f"single_s{strength}",
                0.0,
                MAIN_THRESHOLD,
                "original",
                attacker_id,
            )
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    # Honest reference
    honest_values = [
        get_final_contribution(
            history,
            "clean",
            0.0,
            MAIN_THRESHOLD,
            "original",
            attacker_id,
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print_series_table(
        "FIGURE 1 - Attack effectiveness without defense",
        ATTACK_STRENGTHS,
        means,
        stds,
        "Attack strength",
    )

    print(
        f"Honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    # ------------------------------------------------------------
    # Export PGFPlots data
    # ------------------------------------------------------------

    write_dat(
        "figure1_attack_without_defense.dat",
        ["strength", "mean", "std"],
        [
            (strength, mean, std)
            for strength, mean, std in zip(
                ATTACK_STRENGTHS,
                means,
                stds,
            )
        ],
    )

    write_dat(
        "figure1_attack_without_defense_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Original Python plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    x = np.arange(len(ATTACK_STRENGTHS))

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label="Attacker",
    )

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(s) for s in ATTACK_STRENGTHS]
    )

    ax.set_xlabel(
        "Reported fake self-contribution strength"
    )

    ax.set_ylabel(
        r"Final attacker contribution $\phi_i^{(T)}(i)$"
    )

    ax.set_title(
        "Figure 1: Attack effectiveness without defense (RQ1)"
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    for xi, mean in zip(x, means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(xi, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "figure1_attack_without_defense",
    )


# ================================================================
# Figure 2 (RQ3) - Outlier-only defense against stealth attacks
# ================================================================

def plot_outlier_only_stealth(results):

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    exported = []

    for scenario, label in STEALTH_SCENARIOS:

        means = []
        stds = []

        for threshold in THRESHOLDS:

            values = [
                get_final_contribution(
                    history,
                    scenario,
                    0.0,
                    threshold,
                    "audited",
                    history["single_attacker_id"],
                )
                for history in results
            ]

            mean, std = mean_std(values)

            means.append(mean)
            stds.append(std)

        print_series_table(
            f"FIGURE 2 - {label}",
            THRESHOLDS,
            means,
            stds,
            "Threshold",
        )

        ax.errorbar(
            THRESHOLDS,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=label,
        )

        exported.append(
            (threshold, means, stds)
        )

        # Export separately for each stealth attack
        safe_label = (
            scenario.replace("-", "_")
        )

        write_dat(
            f"figure2_{safe_label}.dat",
            ["threshold", "mean", "std"],
            [
                (threshold, mean, std)
                for threshold, mean, std in zip(
                    THRESHOLDS,
                    means,
                    stds,
                )
            ],
        )

    # ------------------------------------------------------------
    # Honest reference
    # ------------------------------------------------------------

    honest_values = [
        get_final_contribution(
            history,
            "clean",
            0.0,
            MAIN_THRESHOLD,
            "original",
            history["single_attacker_id"],
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print(
        f"Figure 2 honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    write_dat(
        "figure2_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Outlier detection threshold"
    )

    ax.set_ylabel(
        r"Final attacker contribution $\phi_i^{(T)}(i)$"
    )

    ax.set_title(
        "Figure 2: Outlier detection alone against stealth attacks (RQ3)"
    )

    ax.grid(
        True,
        alpha=0.3,
        which="both",
    )

    ax.legend()

    save_figure(
        fig,
        "figure2_outlier_only_stealth",
    )


# ================================================================
# Figure 3 (RQ2) - Honest reports flagged by outlier detection
# ================================================================

def plot_honest_false_positives(results):

    means = []
    stds = []

    for threshold in THRESHOLDS:

        values = [
            get_audit_statistics(
                history,
                "clean",
                0.0,
                threshold,
            )["honest_outlier_audit_rate"]
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    print_series_table(
        "FIGURE 3 - Honest reports flagged by outlier detection",
        THRESHOLDS,
        means,
        stds,
        "Threshold",
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    write_dat(
        "figure3_honest_false_positives.dat",
        ["threshold", "mean", "std"],
        [
            (threshold, mean, std)
            for threshold, mean, std in zip(
                THRESHOLDS,
                means,
                stds,
            )
        ],
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.errorbar(
        THRESHOLDS,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Outlier detection threshold"
    )

    ax.set_ylabel(
        "Fraction of honest reports flagged by outlier detector"
    )

    ax.set_title(
        "Figure 3: Honest reports flagged by outlier detection (RQ2)"
    )

    ax.set_ylim(-0.05, 1.05)

    ax.grid(
        True,
        alpha=0.3,
        which="both",
    )

    for threshold, mean in zip(
        THRESHOLDS,
        means,
    ):
        ax.annotate(
            f"{mean:.3f}",
            xy=(threshold, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "figure3_honest_false_positives",
    )


# ================================================================
# Figure 4 (RQ3) - Effect of probabilistic auditing
# ================================================================

def plot_audit_only_attack(results):

    scenario = "single_s1"

    means = []
    stds = []

    for p in AUDIT_PROBABILITIES:

        values = [
            get_final_contribution(
                history,
                scenario,
                p,
                None,
                "audited",
                history["single_attacker_id"],
            )
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    # ------------------------------------------------------------
    # Honest baseline
    # ------------------------------------------------------------

    honest_values = [
        get_final_contribution(
            history,
            "clean",
            MAIN_AUDIT_PROBABILITY,
            None,
            "audited",
            history["single_attacker_id"],
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print_series_table(
        "FIGURE 4 - Probabilistic auditing",
        AUDIT_PROBABILITIES,
        means,
        stds,
        "Audit probability p",
    )

    print(
        f"Honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    write_dat(
        "figure4_audit_only_attack.dat",
        ["p", "mean", "std"],
        [
            (p, mean, std)
            for p, mean, std in zip(
                AUDIT_PROBABILITIES,
                means,
                stds,
            )
        ],
    )

    write_dat(
        "figure4_audit_only_attack_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.errorbar(
        AUDIT_PROBABILITIES,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label="Attacker (strength 1)",
    )

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_xlabel(
        "Random audit probability $p$"
    )

    ax.set_ylabel(
        r"Final attacker contribution $\phi_i^{(T)}(i)$"
    )

    ax.set_title(
        "Figure 4: Effect of probabilistic auditing on attack reward (RQ3)"
    )

    ax.set_xlim(-0.05, 1.05)

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    for p, mean in zip(
        AUDIT_PROBABILITIES,
        means,
    ):
        ax.annotate(
            f"{mean:.3f}",
            xy=(p, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "figure4_audit_only_attack",
    )


# ================================================================
# Figure 5 (RQ3) - Comparison of defense configurations
# ================================================================

def plot_combined_defense(results):

    threshold = MAIN_THRESHOLD
    p = MAIN_AUDIT_PROBABILITY

    configurations = [
        (
            "No defense",
            f"stealth_fixed_full_{threshold}",
            0.0,
            None,
            "original",
        ),
        (
            "Outlier-only",
            "stealth_full",
            0.0,
            threshold,
            "audited",
        ),
        (
            "Audit-only",
            f"stealth_fixed_full_{threshold}",
            p,
            None,
            "audited",
        ),
        (
            "Combined",
            "stealth_full",
            p,
            threshold,
            "audited",
        ),
    ]

    means = []
    stds = []

    for (
        label,
        scenario,
        audit_probability,
        outlier_threshold,
        version,
    ) in configurations:

        values = [
            get_final_contribution(
                history,
                scenario,
                audit_probability,
                outlier_threshold,
                version,
                history["single_attacker_id"],
            )
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    print_series_table(
        "FIGURE 5 - Comparison of defense configurations",
        [c[0] for c in configurations],
        means,
        stds,
        "Configuration",
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    write_dat(
        "figure5_combined_defense.dat",
        ["configuration", "mean", "std"],
        [
            (label, mean, std)
            for (label, *_), mean, std in zip(
                configurations,
                means,
                stds,
            )
        ],
    )

    # ------------------------------------------------------------
    # Honest baseline
    # ------------------------------------------------------------

    honest_values = [
        get_final_contribution(
            history,
            "clean",
            p,
            threshold,
            "audited",
            history["single_attacker_id"],
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print(
        f"Honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    write_dat(
        "figure5_combined_defense_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x = np.arange(len(configurations))

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [c[0] for c in configurations]
    )

    ax.set_ylabel(
        r"Final attacker contribution $\phi_i^{(T)}(i)$"
    )

    ax.set_title(
        "Figure 5: Comparison of defense configurations (RQ3)"
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    for xi, mean in zip(x, means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(xi, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "figure5_combined_defense",
    )


# ================================================================
# Figure 6 (RQ3) - Byzantine collusion comparison
# ================================================================

def plot_byzantine_attack(results):

    scenario = "byzantine_collusion"

    byzantine_target = 2

    configurations = [
        (
            "No defense",
            "original",
            0.0,
            None,
        ),
        (
            "Outlier-only",
            "audited",
            0.0,
            MAIN_THRESHOLD,
        ),
        (
            "Audit-only",
            "audited",
            MAIN_AUDIT_PROBABILITY,
            None,
        ),
        (
            "Combined",
            "audited",
            MAIN_AUDIT_PROBABILITY,
            MAIN_THRESHOLD,
        ),
    ]

    means = []
    stds = []

    for (
        label,
        version,
        p,
        threshold,
    ) in configurations:

        values = [
            get_final_contribution(
                history,
                scenario,
                p,
                threshold,
                version,
                byzantine_target,
            )
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    print_series_table(
        "FIGURE 6 - Byzantine collusion attack",
        [c[0] for c in configurations],
        means,
        stds,
        "Configuration",
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    write_dat(
        "figure6_byzantine_attack.dat",
        ["configuration", "mean", "std"],
        [
            (label, mean, std)
            for (label, *_), mean, std in zip(
                configurations,
                means,
                stds,
            )
        ],
    )

    # ------------------------------------------------------------
    # Honest baseline
    # ------------------------------------------------------------

    honest_values = [
        get_final_contribution(
            history,
            "clean",
            MAIN_AUDIT_PROBABILITY,
            MAIN_THRESHOLD,
            "audited",
            byzantine_target,
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(
        honest_values
    )

    print(
        f"Honest baseline for client "
        f"{byzantine_target}: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    write_dat(
        "figure6_byzantine_attack_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x = np.arange(
        len(configurations)
    )

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [c[0] for c in configurations]
    )

    ax.set_ylabel(
        r"Final target contribution $\phi_2^{(T)}(2)$"
    )

    ax.set_title(
        "Figure 6: Byzantine collusion attack"
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    for xi, mean in zip(x, means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(xi, mean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "figure6_byzantine_attack",
    )


# ================================================================
# Figure 7 (RQ3) - Interaction between outlier detection and auditing
# ================================================================

def plot_p_tau_interaction(results):

    scenario = "stealth_full"

    mean_matrix = np.zeros(
        (
            len(THRESHOLDS),
            len(AUDIT_PROBABILITIES),
        )
    )

    std_matrix = np.zeros_like(mean_matrix)

    # ------------------------------------------------------------
    # Compute mean and standard deviation for every (tau, p)
    # ------------------------------------------------------------

    for i, threshold in enumerate(THRESHOLDS):

        for j, p in enumerate(AUDIT_PROBABILITIES):

            values = [
                get_final_contribution(
                    history,
                    scenario,
                    p,
                    threshold,
                    "audited",
                    history["single_attacker_id"],
                )
                for history in results
            ]

            mean, std = mean_std(values)

            mean_matrix[i, j] = mean
            std_matrix[i, j] = std

    # ------------------------------------------------------------
    # Print numerical table
    # ------------------------------------------------------------

    print(
        "\nFIGURE 7 - Interaction between outlier detection "
        "and probabilistic auditing"
    )

    print("\nMean final attacker contribution:")

    header = (
        f"{'tau':>10}"
        + "".join(
            f"{p:>12.2f}"
            for p in AUDIT_PROBABILITIES
        )
    )

    print(header)
    print("-" * len(header))

    for i, threshold in enumerate(THRESHOLDS):

        row = (
            f"{threshold:>10.3f}"
            + "".join(
                f"{mean_matrix[i, j]:>12.3f}"
                for j in range(
                    len(AUDIT_PROBABILITIES)
                )
            )
        )

        print(row)

    # ------------------------------------------------------------
    # Honest baseline
    # ------------------------------------------------------------

    honest_values = [
        get_final_contribution(
            history,
            "clean",
            MAIN_AUDIT_PROBABILITY,
            MAIN_THRESHOLD,
            "audited",
            history["single_attacker_id"],
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print(
        f"\nHonest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    # ------------------------------------------------------------
    # Export original long-format data
    #
    # tau p mean std
    # ------------------------------------------------------------

    heatmap_rows = []

    for i, threshold in enumerate(THRESHOLDS):

        for j, p in enumerate(AUDIT_PROBABILITIES):

            heatmap_rows.append(
                (
                    threshold,
                    p,
                    mean_matrix[i, j],
                    std_matrix[i, j],
                )
            )

    write_dat(
        "figure7_p_tau_interaction.dat",
        ["tau", "p", "mean", "std"],
        heatmap_rows,
    )

    # ------------------------------------------------------------
    # Export honest baseline
    # ------------------------------------------------------------

    write_dat(
        "figure7_p_tau_interaction_honest.dat",
        ["mean", "std"],
        [(honest_mean, honest_std)],
    )

    # ------------------------------------------------------------
    # Export heatmap data specifically for PGFPlots
    #
    # This matches the exact structure used by the working
    # inline PGFPlots matrix plot:
    #
    # x y mean
    #
    # x = index of audit probability
    # y = index of threshold
    # mean = measured contribution
    # ------------------------------------------------------------

    heatmap_pgfplots_rows = []

    for i in range(len(THRESHOLDS)):

        for j in range(len(AUDIT_PROBABILITIES)):

            heatmap_pgfplots_rows.append(
                (
                    j,
                    i,
                    mean_matrix[i, j],
                )
            )

    write_dat(
        "figure7_heatmap.dat",
        ["x", "y", "mean"],
        heatmap_pgfplots_rows,
    )

    # ------------------------------------------------------------
    # Export standard deviation in the same PGFPlots format
    # ------------------------------------------------------------

    heatmap_std_rows = []

    for i in range(len(THRESHOLDS)):

        for j in range(len(AUDIT_PROBABILITIES)):

            heatmap_std_rows.append(
                (
                    j,
                    i,
                    std_matrix[i, j],
                )
            )

    write_dat(
        "figure7_heatmap_std.dat",
        ["x", "y", "std"],
        heatmap_std_rows,
    )

    # ------------------------------------------------------------
    # Keep the original wide matrix export as well
    # ------------------------------------------------------------

    write_dat(
        "figure7_p_tau_interaction_matrix.dat",
        ["tau"] + [
            f"p{str(p).replace('.', '_')}"
            for p in AUDIT_PROBABILITIES
        ],
        [
            (
                threshold,
                *[
                    mean_matrix[i, j]
                    for j in range(
                        len(AUDIT_PROBABILITIES)
                    )
                ],
            )
            for i, threshold in enumerate(THRESHOLDS)
        ],
    )

    # ------------------------------------------------------------
    # Export standard deviation matrix
    # ------------------------------------------------------------

    write_dat(
        "figure7_p_tau_interaction_std_matrix.dat",
        ["tau"] + [
            f"p{str(p).replace('.', '_')}"
            for p in AUDIT_PROBABILITIES
        ],
        [
            (
                threshold,
                *[
                    std_matrix[i, j]
                    for j in range(
                        len(AUDIT_PROBABILITIES)
                    )
                ],
            )
            for i, threshold in enumerate(THRESHOLDS)
        ],
    )

    # ------------------------------------------------------------
    # Original Python heatmap
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    image = ax.imshow(
        mean_matrix,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        np.arange(
            len(AUDIT_PROBABILITIES)
        )
    )

    ax.set_xticklabels(
        [
            str(p)
            for p in AUDIT_PROBABILITIES
        ]
    )

    ax.set_yticks(
        np.arange(
            len(THRESHOLDS)
        )
    )

    ax.set_yticklabels(
        [
            str(t)
            for t in THRESHOLDS
        ]
    )

    ax.set_xlabel(
        "Random audit probability $p$"
    )

    ax.set_ylabel(
        r"Outlier detection threshold $\tau$"
    )

    ax.set_title(
        "Figure 7: Interaction between outlier detection "
        "and probabilistic auditing (RQ3)"
    )

    # ------------------------------------------------------------
    # Numerical values inside heatmap
    # ------------------------------------------------------------

    for i in range(
        len(THRESHOLDS)
    ):

        for j in range(
            len(AUDIT_PROBABILITIES)
        ):

            ax.text(
                j,
                i,
                f"{mean_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
            )

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"Mean final attacker contribution "
        r"$\phi_i^{(T)}(i)$"
    )

    ax.grid(False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure7_p_tau_interaction",
    )

# ================================================================
# Tables (printed only - closed-form / confirmatory, not worth a
# figure each)
# ================================================================

def print_computational_overhead_table():

    print("\n" + "=" * 70)
    print("TABLE: Computational overhead (theoretical, C = 1+p)")
    print("=" * 70)

    for p in AUDIT_PROBABILITIES:
        print(f"  p={p:<5} relative LCV cost = {1.0 + p:.2f}x")


def print_theoretical_threshold_table(results):

    print("\n" + "=" * 70)
    print("TABLE: Theoretical threshold reference values")
    print("(t <= p*e / (n*(1-p)) - reference only, not enforced)")
    print("=" * 70)

    for i, history in enumerate(results):

        thresholds = history.get("theoretical_thresholds")
        e_estimate = history.get("honest_reward_estimate")
        audit_threshold = history.get("audit_threshold")

        if not thresholds:
            print(f"\nSeed {SEEDS[i]}: no theoretical_thresholds saved.")
            continue

        print(
            f"\nSeed {SEEDS[i]}: e={e_estimate:.6f}, "
            f"actual audit_threshold used={audit_threshold:.2e}"
        )

        for p, variants in thresholds.items():

            with_n = variants["with_neighbors"]
            without_n = variants["without_neighbors"]

            with_n_str = (
                f"{with_n:.6f}" if with_n is not None else "vacuous"
            )
            without_n_str = (
                f"{without_n:.6f}"
                if without_n is not None else "vacuous"
            )

            print(
                f"  p={p:<5} t_with_neighbors={with_n_str:<12} "
                f"t_without_neighbors={without_n_str}"
            )


def print_clean_false_positive_table(results):

    print("\n" + "=" * 70)
    print("TABLE: Clean-scenario false-positive rates (sanity check)")
    print("=" * 70)

    for threshold in [None] + THRESHOLDS:
        for p in [0.0, MAIN_AUDIT_PROBABILITY]:

            values = [
                get_audit_statistics(
                    history, "clean", p, threshold,
                )["false_positive_rate"]
                for history in results
            ]

            mean, std = mean_std(values)

            print(
                f"  p={p:<5} threshold={str(threshold):<8} "
                f"false_positive_rate={mean:.4f} (+/- {std:.4f})"
            )


# ================================================================
# Main
# ================================================================

# ================================================================
# Main
# ================================================================

def main():

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
    })

    results = load_all_results()

    print("\n" + "=" * 70)
    print("Generating figures")
    print("=" * 70)

    print("\n[1/5] Attack without defense")
    plot_attack_without_defense(results)

    print("\n[2/5] Outlier-only defense against stealth attacks")
    plot_outlier_only_stealth(results)

    print("\n[3/5] Honest reports flagged by outlier detection")
    plot_honest_false_positives(results)

    print("\n[4/5] Effect of probabilistic auditing")
    plot_audit_only_attack(results)

    print("\n[5/5] Combined defense")
    plot_combined_defense(results)

    plot_byzantine_attack(results)

    plot_p_tau_interaction(results)

    print("\n" + "=" * 70)
    print("All figures generated.")
    print(f"Output directory: {FIGURES_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()