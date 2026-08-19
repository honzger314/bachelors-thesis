import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Configuration
# ================================================================

RESULTS_DIR = Path("results3")
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
THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.10]
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
# Figure 1 (RQ1) - Attack works without any defense
# ================================================================

def plot_attack_without_defense(results):
    """
    Shows that the attacker can inflate its final self-contribution
    when neither outlier detection nor auditing is active.

    X-axis:
        attack strength

    Y-axis:
        final attacker self-contribution

    Configuration:
        original
        p = 0
        threshold = MAIN_THRESHOLD

    The threshold value is irrelevant here because the original
    mechanism does not apply the defense. We nevertheless use the
    same stored scenario key consistently.
    """

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
        f"Honest baseline: {honest_mean:.6f} ± {honest_std:.6f}"
    )

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
    """
    Shows the attacker's final contribution against calibrated
    stealth attacks when only the statistical outlier detector is
    active.

    p = 0
    version = audited

    The stealth attack is calibrated for each threshold. Therefore
    the plot asks:

        "Can the attacker still obtain a large contribution while
         staying below the detector's threshold?"

    Both stealth_half and stealth_full are shown.
    """

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

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

    # Honest reference at the representative threshold
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

    honest_mean, _ = mean_std(honest_values)

    honest_mean, honest_std = mean_std(honest_values)

    print(
        f"Figure 2 honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

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
        "Figure 2: Outlier detection alone against stealth attacks "
        "(RQ3)"
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
    """
    Shows how often honest reports are flagged by the statistical
    outlier detector as the detection threshold changes.

    This is specifically the OUTLIER detector's false-positive
    behavior.

    p = 0 isolates the outlier detector from probabilistic auditing.

    The measured quantity is:

        honest_outlier_audit_rate

    i.e. the fraction of honest reports that were sent to audit
    because they were flagged as outliers.

    This is different from false_positive_rate, which measures
    reports that were ultimately rejected by the audit verification.
    """

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
    """
    Shows how probabilistic auditing reduces the attacker's final
    contribution as the audit probability p increases.

    Outlier detection is DISABLED:
        threshold = None

    This isolates the effect of probabilistic auditing itself.

    Attack:
        single_s1

    X-axis:
        random audit probability p

    Y-axis:
        final attacker self-contribution
    """

    scenario = "single_s1"

    means = []
    stds = []

    for p in AUDIT_PROBABILITIES:

        values = []

        for history in results:

            attacker_id = history["single_attacker_id"]

            values.append(
                get_final_contribution(
                    history,
                    scenario,
                    p,
                    None,          # disable outlier detection
                    "audited",
                    attacker_id,
                )
            )

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
            None,              # no outlier detection
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
        f"Honest baseline: {honest_mean:.6f} ± {honest_std:.6f}"
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
        "Figure 4: Effect of probabilistic auditing on attack reward "
        "(RQ3)"
    )

    ax.set_xlim(-0.05, 1.05)

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    # ------------------------------------------------------------
    # Annotate values
    # ------------------------------------------------------------

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
    """
    Compares the four defense configurations under the same fixed
    attack.

    Attack:
        single_s1

    Representative operating point:
        p = MAIN_AUDIT_PROBABILITY
        threshold = MAIN_THRESHOLD

    Configurations:
        1. No defense
        2. Outlier-only
        3. Audit-only
        4. Combined

    The same attack is used in every configuration so that the
    comparison isolates the effect of the defense mechanism.
    """

    scenario = "single_s1"

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

    # ------------------------------------------------------------
    # Attacker contribution for each configuration
    # ------------------------------------------------------------

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
                history["single_attacker_id"],
            )
            for history in results
        ]

        mean, std = mean_std(values)

        means.append(mean)
        stds.append(std)

    # ------------------------------------------------------------
    # Print numerical values
    # ------------------------------------------------------------

    print_series_table(
        "FIGURE 5 - Defense configuration comparison",
        [c[0] for c in configurations],
        means,
        stds,
        "Configuration",
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
            history["single_attacker_id"],
        )
        for history in results
    ]

    honest_mean, honest_std = mean_std(honest_values)

    print(
        f"Honest baseline: "
        f"{honest_mean:.6f} ± {honest_std:.6f}"
    )

    print(
        f"Operating point: "
        f"p={MAIN_AUDIT_PROBABILITY}, "
        f"threshold={MAIN_THRESHOLD}"
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
        "Figure 5: Comparison of defense configurations (RQ3)\n"
        f"Fixed attack (strength 1), "
        f"p={MAIN_AUDIT_PROBABILITY}, "
        f"threshold={MAIN_THRESHOLD}"
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend()

    # ------------------------------------------------------------
    # Value labels
    # ------------------------------------------------------------

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

    print("\n" + "=" * 70)
    print("All figures generated.")
    print(f"Output directory: {FIGURES_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()