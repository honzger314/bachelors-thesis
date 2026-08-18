import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Configuration
# ================================================================

RESULTS_DIR = Path("results2")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [1, 2, 3]

TOPOLOGY = "ring"
NUM_CLIENTS = 10
ROUNDS = 20

# Main operating point used by the proposed defense
MAIN_AUDIT_PROBABILITY = 0.20
MAIN_THRESHOLD = 0.01

# Experimental values
AUDIT_PROBABILITIES = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
]

THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.10,
]

ATTACK_STRENGTHS = [1, 5, 10, 20, 50]


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
# Scenario helpers
# ================================================================

def scenario_key(base_scenario, audit_probability, threshold):
    """
    Construct the scenario key used by the simulator.
    """

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
# Contribution helpers
# ================================================================

def get_contribution_entry(
    history,
    base_scenario,
    audit_probability,
    threshold,
    round_idx,
    version,
):
    """
    Get one contribution matrix.
    """

    scenario = scenario_key(
        base_scenario,
        audit_probability,
        threshold,
    )

    contributions = history["contributions"][scenario]

    entry = contributions[round_idx][version]

    client_ids = sorted(entry.keys())

    matrix = np.stack(
        [
            np.asarray(entry[cid], dtype=float)
            for cid in client_ids
        ]
    )

    return matrix, client_ids


def get_final_contribution(
    history,
    base_scenario,
    audit_probability,
    threshold,
    version,
    attacker_id,
):
    """
    Return the attacker's self-contribution in the final round.
    """

    matrix, client_ids = get_contribution_entry(
        history,
        base_scenario,
        audit_probability,
        threshold,
        ROUNDS - 1,
        version,
    )

    idx = client_ids.index(attacker_id)

    return float(matrix[idx, idx])


# ================================================================
# Audit statistics
# ================================================================

def get_audit_statistics(
    history,
    base_scenario,
    audit_probability,
    threshold,
):
    """
    Calculate audit statistics for one experiment.

    Important distinction:

        random_audit
            selected because of probabilistic auditing

        outlier_audit
            selected because of outlier detection

        flagged
            actually identified as malicious

    We keep these separate because Graph 4 specifically studies
    the trade-off caused by threshold-based outlier detection.
    """

    audit_logs = history.get("audit_logs", {})

    key = scenario_key(
        base_scenario,
        audit_probability,
        threshold,
    )

    if key not in audit_logs:
        return {
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

    attacker_ids = set()

    single_attacker = history.get(
        "single_attacker_id"
    )

    if single_attacker is not None:
        attacker_ids.add(single_attacker)

    attacker_ids.update(
        history.get("multi_attacker_ids", [])
    )

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

    log = audit_logs[key]

    for round_log in log:

        if not isinstance(round_log, dict):
            continue

        for client_id, outcome in round_log.items():

            if not isinstance(outcome, dict):
                continue

            total_reports += 1

            audited = bool(
                outcome.get("audited", False)
            )

            random_audit = bool(
                outcome.get("random_audit", False)
            )

            outlier_audit = bool(
                outcome.get("outlier_audit", False)
            )

            flagged = bool(
                outcome.get("flagged", False)
            )

            if audited:
                total_audited += 1

            if random_audit:
                total_random_audits += 1

            if outlier_audit:
                total_outlier_audits += 1

            is_malicious = (
                client_id in attacker_ids
            )

            if is_malicious:

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

    return {
        "audit_rate": (
            total_audited / total_reports
            if total_reports > 0
            else np.nan
        ),

        "malicious_audit_rate": (
            malicious_audited / malicious_reports
            if malicious_reports > 0
            else np.nan
        ),

        "malicious_detection_rate": (
            malicious_detected / malicious_reports
            if malicious_reports > 0
            else np.nan
        ),

        "honest_audit_rate": (
            honest_audited / honest_reports
            if honest_reports > 0
            else np.nan
        ),

        "false_positive_rate": (
            honest_flagged / honest_reports
            if honest_reports > 0
            else np.nan
        ),

        "random_audit_rate": (
            total_random_audits / total_reports
            if total_reports > 0
            else np.nan
        ),

        "outlier_audit_rate": (
            total_outlier_audits / total_reports
            if total_reports > 0
            else np.nan
        ),

        "malicious_outlier_audit_rate": (
            malicious_outlier_audits / malicious_reports
            if malicious_reports > 0
            else np.nan
        ),

        "honest_outlier_audit_rate": (
            honest_outlier_audits / honest_reports
            if honest_reports > 0
            else np.nan
        ),
    }


# ================================================================
# Generic statistics
# ================================================================

def mean_std(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(np.nanmean(values)),
        float(np.nanstd(values)),
    )


def save_figure(fig, filename):
    pdf_path = FIGURES_DIR / f"{filename}.pdf"
    png_path = FIGURES_DIR / f"{filename}.png"

    fig.tight_layout()

    fig.savefig(pdf_path)

    fig.savefig(
        png_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


# ================================================================
# Plot 1
#
# Attack effectiveness
# ================================================================

def plot_attack_effectiveness(results):
    """
    Shows that contribution inflation attacks are effective
    against the original method and how the modified method
    limits this effect.
    """

    attacker_id = results[0][
        "single_attacker_id"
    ]

    original = {
        s: []
        for s in ATTACK_STRENGTHS
    }

    modified = {
        s: []
        for s in ATTACK_STRENGTHS
    }

    honest = []

    for history in results:

        honest.append(
            get_final_contribution(
                history,
                "clean",
                MAIN_AUDIT_PROBABILITY,
                MAIN_THRESHOLD,
                "original",
                attacker_id,
            )
        )

        for strength in ATTACK_STRENGTHS:

            scenario = (
                f"single_s{strength}"
            )

            original[strength].append(
                get_final_contribution(
                    history,
                    scenario,
                    MAIN_AUDIT_PROBABILITY,
                    MAIN_THRESHOLD,
                    "original",
                    attacker_id,
                )
            )

            modified[strength].append(
                get_final_contribution(
                    history,
                    scenario,
                    MAIN_AUDIT_PROBABILITY,
                    MAIN_THRESHOLD,
                    "modified",
                    attacker_id,
                )
            )

    original_mean = np.array([
        np.mean(original[s])
        for s in ATTACK_STRENGTHS
    ])

    modified_mean = np.array([
        np.mean(modified[s])
        for s in ATTACK_STRENGTHS
    ])

    honest_mean = np.mean(honest)

    x = np.arange(
        len(ATTACK_STRENGTHS)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.plot(
        x,
        original_mean,
        marker="o",
        label="Original TRIP-Shapley",
    )

    ax.plot(
        x,
        modified_mean,
        marker="o",
        label="Modified TRIP-Shapley",
    )

    ax.axhline(
        honest_mean,
        linestyle=":",
        label="Honest baseline",
    )

    ax.set_yscale(
        "symlog",
        linthresh=0.1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            str(s)
            for s in ATTACK_STRENGTHS
        ]
    )

    ax.set_xlabel(
        "Reported fake self-contribution strength"
    )

    ax.set_ylabel(
        r"Final attacker contribution "
        r"$\phi_i^{(T)}(i)$"
    )

    ax.set_title(
        "Attack effectiveness"
        f" (p={MAIN_AUDIT_PROBABILITY}, "
        f"threshold={MAIN_THRESHOLD})"
    )

    ax.grid(
        True,
        alpha=0.3,
        which="both",
    )

    ax.legend()

    save_figure(
        fig,
        "plot1_attack_effectiveness",
    )


# ================================================================
# Plot 2
#
# Detection vs random audit probability
# ================================================================

def plot_detection_vs_audit_probability(
    results,
):
    """
    Shows that random auditing detects malicious clients
    at approximately the configured probability p.

    Outlier detection is disabled.
    """

    detection_means = []
    detection_stds = []

    for p in AUDIT_PROBABILITIES:

        values = []

        for history in results:

            stats = get_audit_statistics(
                history,
                "single_s1",
                p,
                None,
            )

            values.append(
                stats[
                    "malicious_audit_rate"
                ]
            )

        mean, std = mean_std(
            values
        )

        detection_means.append(mean)
        detection_stds.append(std)

    p_values = np.asarray(
        AUDIT_PROBABILITIES,
        dtype=float,
    )

    detection_means = np.asarray(
        detection_means
    )

    detection_stds = np.asarray(
        detection_stds
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.errorbar(
        p_values,
        detection_means,
        yerr=detection_stds,
        marker="o",
        capsize=4,
        label="Measured detection rate",
    )

    ax.plot(
        p_values,
        p_values,
        linestyle="--",
        label="Ideal random-audit rate",
    )

    ax.set_xlabel(
        "Random audit probability $p$"
    )

    ax.set_ylabel(
        "Malicious reports audited"
    )

    ax.set_title(
        "Probabilistic auditing detects malicious clients"
    )

    ax.set_xlim(
        0,
        1.05,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_figure(
        fig,
        "plot2_detection_vs_audit_probability",
    )

    print(
        "\nRandom-audit detection results:"
    )

    for p, mean, std in zip(
        p_values,
        detection_means,
        detection_stds,
    ):
        print(
            f"  p={p:.2f}: "
            f"detection={mean:.3f} "
            f"± {std:.3f}"
        )


# ================================================================
# Plot 3
#
# Audit probability vs computation cost
# ================================================================

def plot_audit_cost(results):
    """
    Shows the theoretical computational overhead of probabilistic
    auditing.

    Baseline:
        N LCV computations.

    Random auditing:
        approximately p*N additional LCV computations.

    Therefore:
        relative cost = 1 + p
    """

    p_values = np.asarray(
        AUDIT_PROBABILITIES,
        dtype=float,
    )

    relative_cost = (
        1.0 + p_values
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.plot(
        p_values,
        relative_cost,
        linestyle="--",
        label=r"Theoretical cost: $1+p$",
    )

    ax.scatter(
        p_values,
        relative_cost,
        s=55,
        zorder=3,
        label="Configured audit probabilities",
    )

    for p, cost in zip(
        p_values,
        relative_cost,
    ):
        ax.annotate(
            f"p={p:g}",
            xy=(p, cost),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(
        "Random audit probability $p$"
    )

    ax.set_ylabel(
        "Relative LCV computation cost"
    )

    ax.set_title(
        "Computational overhead of probabilistic auditing"
    )

    ax.set_xlim(
        0,
        1.05,
    )

    ax.set_ylim(
        0.95,
        2.05,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_figure(
        fig,
        "plot3_audit_cost_tradeoff",
    )

    print("\nAudit cost:")

    for p, cost in zip(
        p_values,
        relative_cost,
    ):
        print(
            f"  p={p:.2f}: "
            f"relative LCV cost={cost:.3f}"
        )


# ================================================================
# Plot 4
#
# Threshold-only detection trade-off
# ================================================================

def plot_threshold_tradeoff(results):
    """
    Shows the fundamental problem with relying on outlier detection
    alone.

    Left axis:
        fraction of stealth attacks detected by outlier detection.

    Right axis:
        fraction of honest reports selected for outlier auditing.

    As the threshold becomes larger:
        - stealth detection becomes weaker
        - honest-client auditing decreases

    This demonstrates why choosing a threshold alone creates a
    security/computation trade-off.
    """

    scenarios = [
        (
            "stealth_half",
            "Stealth half",
        ),
        (
            "stealth_full",
            "Stealth full",
        ),
    ]

    threshold_values = np.asarray(
        THRESHOLDS,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Stealth detection
    # ------------------------------------------------------------

    detection_by_scenario = {}

    for scenario, label in scenarios:

        means = []
        stds = []

        for threshold in threshold_values:

            values = []

            for history in results:

                stats = get_audit_statistics(
                    history,
                    scenario,
                    MAIN_AUDIT_PROBABILITY,
                    threshold,
                )

                values.append(
                    stats[
                        "malicious_outlier_audit_rate"
                    ]
                )

            mean, std = mean_std(
                values
            )

            means.append(mean)
            stds.append(std)

        detection_by_scenario[
            scenario
        ] = (
            np.asarray(means),
            np.asarray(stds),
        )

    # ------------------------------------------------------------
    # Honest-client outlier audit rate
    #
    # Use clean experiments because there are no malicious clients
    # in those runs. Therefore every outlier audit is necessarily
    # directed at an honest client.
    # ------------------------------------------------------------

    honest_means = []
    honest_stds = []

    for threshold in threshold_values:

        values = []

        for history in results:

            stats = get_audit_statistics(
                history,
                "clean",
                MAIN_AUDIT_PROBABILITY,
                threshold,
            )

            values.append(
                stats[
                    "outlier_audit_rate"
                ]
            )

        mean, std = mean_std(
            values
        )

        honest_means.append(mean)
        honest_stds.append(std)

    honest_means = np.asarray(
        honest_means
    )

    honest_stds = np.asarray(
        honest_stds
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(7.5, 5.0)
    )

    # Stealth detection curves
    for scenario, label in scenarios:

        means, stds = (
            detection_by_scenario[
                scenario
            ]
        )

        ax1.errorbar(
            threshold_values,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=f"{label}: attack detection",
        )

    ax1.set_xscale("log")

    ax1.set_xlabel(
        "Outlier detection threshold"
    )

    ax1.set_ylabel(
        "Stealth attacks detected by outlier detector"
    )

    ax1.set_ylim(
        0,
        1.0,
    )

    # Honest-client cost on second axis
    ax2 = ax1.twinx()

    ax2.errorbar(
        threshold_values,
        honest_means,
        yerr=honest_stds,
        marker="s",
        linestyle="--",
        capsize=3,
        label="Honest reports selected for outlier auditing",
    )

    ax2.set_ylabel(
        "Honest reports selected for outlier auditing"
    )

    ax2.set_ylim(
        0,
        1.0,
    )

    ax1.set_title(
        "Threshold-only defense: security vs. overhead"
    )

    ax1.grid(
        True,
        alpha=0.3,
        which="both",
    )

    # Combine legends from both axes.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=8.5,
        loc="best",
    )

    save_figure(
        fig,
        "plot4_threshold_tradeoff",
    )

    # ------------------------------------------------------------
    # Print values
    # ------------------------------------------------------------

    print(
        "\nThreshold-only trade-off:"
    )

    for i, threshold in enumerate(
        threshold_values
    ):

        print(
            f"\n  threshold={threshold:g}"
        )

        for scenario, label in scenarios:

            means, stds = (
                detection_by_scenario[
                    scenario
                ]
            )

            print(
                f"    {label}: "
                f"stealth detection="
                f"{means[i]:.3f} "
                f"± {stds[i]:.3f}"
            )

        print(
            f"    honest outlier audit rate="
            f"{honest_means[i]:.3f} "
            f"± {honest_stds[i]:.3f}"
        )


# ================================================================
# Plot 5
#
# Combined defense: probability of catching a stealth attacker
# ================================================================

def plot_combined_detection(results):
    """
    Shows why probabilistic auditing complements threshold-based
    outlier detection.

    For a malicious client, if random auditing occurs independently
    with probability p per round, the probability of being audited
    at least once after T rounds is:

        1 - (1-p)^T

    We compare this theoretical cumulative probability with the
    experimentally observed probability that the attacker is
    randomly audited at least once.

    The main operating point p=0.20 is highlighted.
    """

    p_values = np.asarray(
        AUDIT_PROBABILITIES,
        dtype=float,
    )

    # Theoretical probability of at least one random audit
    # during the full experiment.
    theoretical_detection = (
        1.0
        - (1.0 - p_values) ** ROUNDS
    )

    # ------------------------------------------------------------
    # Experimental probability of at least one random audit
    # ------------------------------------------------------------

    experimental_means = []
    experimental_stds = []

    for p in p_values:

        per_seed_values = []

        for history in results:

            key = scenario_key(
                "stealth_half",
                p,
                MAIN_THRESHOLD,
            )

            audit_logs = history.get(
                "audit_logs",
                {},
            )

            if key not in audit_logs:
                per_seed_values.append(
                    np.nan
                )
                continue

            attacker_id = history[
                "single_attacker_id"
            ]

            was_audited = False

            for round_log in audit_logs[key]:

                if not isinstance(
                    round_log,
                    dict,
                ):
                    continue

                outcome = round_log.get(
                    attacker_id
                )

                if not isinstance(
                    outcome,
                    dict,
                ):
                    continue

                if outcome.get(
                    "random_audit",
                    False,
                ):
                    was_audited = True
                    break

            per_seed_values.append(
                float(was_audited)
            )

        mean, std = mean_std(
            per_seed_values
        )

        experimental_means.append(
            mean
        )

        experimental_stds.append(
            std
        )

    experimental_means = np.asarray(
        experimental_means
    )

    experimental_stds = np.asarray(
        experimental_stds
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4.8)
    )

    ax.plot(
        p_values,
        theoretical_detection,
        linestyle="--",
        label=(
            r"Theoretical probability of "
            r"at least one audit: "
            r"$1-(1-p)^{20}$"
        ),
    )

    ax.errorbar(
        p_values,
        experimental_means,
        yerr=experimental_stds,
        marker="o",
        capsize=4,
        label="Observed random-audit detection",
    )

    # Highlight main operating point
    main_theoretical = (
        1.0
        - (
            1.0
            - MAIN_AUDIT_PROBABILITY
        ) ** ROUNDS
    )

    ax.scatter(
        [MAIN_AUDIT_PROBABILITY],
        [main_theoretical],
        s=75,
        zorder=4,
        label=(
            f"Chosen setting: p={MAIN_AUDIT_PROBABILITY:g}"
        ),
    )

    ax.annotate(
        f"{main_theoretical:.1%}",
        xy=(
            MAIN_AUDIT_PROBABILITY,
            main_theoretical,
        ),
        xytext=(8, -15),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_xlabel(
        "Random audit probability $p$"
    )

    ax.set_ylabel(
        "Probability attacker is audited at least once"
    )

    ax.set_title(
        f"Cumulative protection from probabilistic auditing "
        f"over {ROUNDS} rounds"
    )

    ax.set_xlim(
        0,
        1.05,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8.5,
    )

    save_figure(
        fig,
        "plot5_combined_defense",
    )

    print(
        "\nCumulative random-audit detection:"
    )

    for p, theoretical, observed, std in zip(
        p_values,
        theoretical_detection,
        experimental_means,
        experimental_stds,
    ):
        print(
            f"  p={p:.2f}: "
            f"theoretical={theoretical:.3f}, "
            f"observed={observed:.3f} "
            f"± {std:.3f}"
        )


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

    print(
        "\n[1/5] Attack effectiveness"
    )
    plot_attack_effectiveness(
        results
    )

    print(
        "\n[2/5] Detection vs random audit probability"
    )
    plot_detection_vs_audit_probability(
        results
    )

    print(
        "\n[3/5] Audit cost trade-off"
    )
    plot_audit_cost(
        results
    )

    print(
        "\n[4/5] Threshold-only security/overhead trade-off"
    )
    plot_threshold_tradeoff(
        results
    )

    print(
        "\n[5/5] Combined defense"
    )
    plot_combined_detection(
        results
    )

    print("\n" + "=" * 70)
    print("All figures generated.")
    print(
        f"Output directory: "
        f"{FIGURES_DIR.resolve()}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()