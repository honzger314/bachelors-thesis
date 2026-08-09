import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# --- Experiment configuration (adjust to match your actual runs) ---

SEEDS = [1, 2, 3]

TOPOLOGY = "ring"
NUM_CLIENTS = 10
ROUNDS = 20

# Which single-attacker strength scenario to use for Plot 2.
# Kept at strength 1.0 to match the multi-attacker scenario's fixed
# strength, so Plot 2 and Plot 3 are directly comparable.
SINGLE_ATTACKER_SCENARIO = "single_s1"

MULTI_ATTACKER_SCENARIOS = ["multi_2", "multi_3"]

HONEST_SCENARIO = "clean"


def result_path(seed):
    return RESULTS_DIR / f"CIFAR10_{TOPOLOGY}_{NUM_CLIENTS}clients_{ROUNDS}rounds_seed_{seed}.pkl"


def load_result(path):
    print(f"Loading: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Update SEEDS / result_path() to match "
            f"your actual filenames."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_seeds():
    histories = []
    for seed in SEEDS:
        h = load_result(result_path(seed))
        print(f"  scenarios found: {list(h['contributions'].keys())}")
        histories.append(h)
    return histories


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def get_round_matrix(history, scenario_name, version, round_idx):
    """
    Returns the contribution matrix for a given scenario, version
    ("original" or "modified"), and round index (0-based).
    """

    contributions = history["contributions"][scenario_name]
    entry = contributions[round_idx][version]
    client_ids = sorted(entry.keys())

    matrix = np.stack(
        [np.asarray(entry[cid], dtype=float) for cid in client_ids]
    )

    return matrix, client_ids


def num_rounds_recorded(history, scenario_name):
    return len(history["contributions"][scenario_name])


# ---------------------------------------------------------------------
# Plot 1: Honest scenario, Spearman rank correlation original vs modified
# ---------------------------------------------------------------------

def spearman_per_round(history, scenario_name=HONEST_SCENARIO):
    """
    For each round, computes the Spearman correlation between the
    original and modified contribution rows for every receiving
    client, then averages across receivers.

    Returns an array of shape (num_rounds,).
    """

    n_rounds = num_rounds_recorded(history, scenario_name)
    rhos = np.full(n_rounds, np.nan)

    for r in range(n_rounds):

        orig_matrix, client_ids = get_round_matrix(
            history, scenario_name, "original", r
        )
        mod_matrix, _ = get_round_matrix(
            history, scenario_name, "modified", r
        )

        row_rhos = []

        for row in range(len(client_ids)):

            rho, _ = spearmanr(orig_matrix[row], mod_matrix[row])

            if np.isfinite(rho):
                row_rhos.append(rho)

        if row_rhos:
            rhos[r] = np.mean(row_rhos)

    return rhos


def plot_honest_ranking_correlation(histories):
    """
    Plot 1: Spearman correlation between original and modified
    contribution rankings in the honest scenario, per round,
    averaged over seeds (mean +/- std band).
    """

    per_seed_rhos = np.stack(
        [spearman_per_round(h, HONEST_SCENARIO) for h in histories]
    )  # shape (n_seeds, n_rounds)

    mean_rho = np.nanmean(per_seed_rhos, axis=0)
    std_rho = np.nanstd(per_seed_rhos, axis=0)

    rounds = np.arange(1, per_seed_rhos.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(rounds, mean_rho, marker="o", color="C0", label=r"$\rho_t$")
    ax.fill_between(
        rounds,
        mean_rho - std_rho,
        mean_rho + std_rho,
        color="C0",
        alpha=0.2,
        label="±1 std (seeds)",
    )

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel(r"Spearman correlation $\rho_t$")
    ax.set_title("Honest scenario: ranking correlation\n(Original vs. Modified TRIP-Shapley)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot1_honest_ranking_correlation.pdf")
    fig.savefig(FIGURES_DIR / "plot1_honest_ranking_correlation.png", dpi=200)
    plt.close(fig)

    print(f"\nPlot 1 saved. Final-round mean rho: {mean_rho[-1]:.4f} (+/- {std_rho[-1]:.4f})")


# ---------------------------------------------------------------------
# Plots 2 & 3: self-contribution of malicious client(s) over rounds
# ---------------------------------------------------------------------

def self_contribution_per_round(history, scenario_name, version, attacker_ids):
    """
    Sums the diagonal entries (self-contribution) of the given
    attacker_ids, for every round, under a given version
    ("original" or "modified").

    Returns an array of shape (num_rounds,).
    """

    n_rounds = num_rounds_recorded(history, scenario_name)
    values = np.zeros(n_rounds)

    for r in range(n_rounds):

        matrix, client_ids = get_round_matrix(
            history, scenario_name, version, r
        )

        idxs = [client_ids.index(cid) for cid in attacker_ids]

        values[r] = sum(matrix[i, i] for i in idxs)

    return values


def _plot_attacker_contribution(
    histories,
    scenario_name,
    attacker_ids_fn,
    title,
    ylabel,
    filename,
):
    """
    Shared plotting logic for Plot 2 (single attacker) and Plot 3
    (aggregated multiple attackers). attacker_ids_fn(history) returns
    the list of malicious client ids to use for that seed's history
    (attacker ids may differ per seed since they were sampled with
    that seed's RNG state).
    """

    per_seed = {"original": [], "modified": []}

    for h in histories:

        attacker_ids = attacker_ids_fn(h)

        for version in ["original", "modified"]:

            series = self_contribution_per_round(
                h, scenario_name, version, attacker_ids
            )

            per_seed[version].append(series)

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {"original": "C3", "modified": "C0"}
    labels = {"original": "Original TRIP-Shapley", "modified": "Modified TRIP-Shapley"}

    rounds = None

    for version in ["original", "modified"]:

        stacked = np.stack(per_seed[version])  # (n_seeds, n_rounds)

        mean_series = np.mean(stacked, axis=0)
        std_series = np.std(stacked, axis=0)

        if rounds is None:
            rounds = np.arange(1, stacked.shape[1] + 1)

        ax.plot(
            rounds, mean_series, marker="o",
            color=colors[version], label=labels[version],
        )
        ax.fill_between(
            rounds,
            mean_series - std_series,
            mean_series + std_series,
            color=colors[version],
            alpha=0.2,
        )

        print(
            f"[{scenario_name} / {version}] final round: "
            f"{mean_series[-1]:.4f} (+/- {std_series[-1]:.4f})"
        )

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{filename}.pdf")
    fig.savefig(FIGURES_DIR / f"{filename}.png", dpi=200)
    plt.close(fig)

    print(f"Saved {filename}.pdf / .png\n")


def self_contribution_final(history, scenario_name, version, attacker_ids):
    """
    Convenience wrapper: final-round value of
    self_contribution_per_round.
    """

    series = self_contribution_per_round(
        history, scenario_name, version, attacker_ids
    )

    return series[-1]


def plot_single_attacker_strength_sweep(histories):
    """
    Plot 2 (central figure): attacker's final contribution as a
    function of attack strength, for both Original and Modified
    TRIP-Shapley, against an honest baseline.

    The honest baseline is the attacker's contribution under the
    ORIGINAL method in the clean (no attack) scenario - i.e. the
    "true" deserved contribution the attacker would get if it
    reported honestly. It is not taken from the modified method,
    since modified always zeroes the self-entry regardless of
    attack presence, so an honest-vs-attacked comparison under
    modified alone would be close to tautological.

    Because the modified method always sets the self-contribution
    entry to zero before propagation - independent of the reported
    strength - its curve is expected to stay flat across strengths,
    while the original method's curve should grow with strength.
    This is the plot that visually demonstrates that vulnerability
    and the fix.
    """

    strengths = [1, 5, 10, 20, 50]
    scenario_names = {s: f"single_s{s}" for s in strengths}

    honest_vals = []
    original_vals = {s: [] for s in strengths}
    modified_vals = {s: [] for s in strengths}

    for h in histories:

        attacker_ids = [h["single_attacker_id"]]

        honest_vals.append(
            self_contribution_final(h, HONEST_SCENARIO, "original", attacker_ids)
        )

        for s in strengths:
            name = scenario_names[s]

            original_vals[s].append(
                self_contribution_final(h, name, "original", attacker_ids)
            )
            modified_vals[s].append(
                self_contribution_final(h, name, "modified", attacker_ids)
            )

    honest_mean = float(np.mean(honest_vals))
    honest_std = float(np.std(honest_vals))

    original_means = np.array([np.mean(original_vals[s]) for s in strengths])
    original_stds = np.array([np.std(original_vals[s]) for s in strengths])

    modified_means = np.array([np.mean(modified_vals[s]) for s in strengths])
    modified_stds = np.array([np.std(modified_vals[s]) for s in strengths])

    x = np.arange(len(strengths))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(
        honest_mean, color="gray", linestyle="--",
        label="Honest baseline (no attack, Original)",
    )
    ax.fill_between(
        x, honest_mean - honest_std, honest_mean + honest_std,
        color="gray", alpha=0.15,
    )

    ax.errorbar(
        x, original_means, yerr=original_stds,
        marker="o", color="C3", capsize=3,
        label="Original TRIP-Shapley",
    )
    ax.errorbar(
        x, modified_means, yerr=modified_stds,
        marker="o", color="C0", capsize=3,
        label="Modified TRIP-Shapley",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in strengths])
    ax.set_xlabel("Reported fake self-contribution strength")
    ax.set_ylabel(r"Attacker final contribution $\phi_i^{(T)}(i)$")
    ax.set_title("Single attacker: final contribution vs. attack strength")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot2_single_attacker_strength_sweep.pdf")
    fig.savefig(FIGURES_DIR / "plot2_single_attacker_strength_sweep.png", dpi=200)
    plt.close(fig)

    print(f"\nHonest baseline (Original, no attack): {honest_mean:.4f} (+/- {honest_std:.4f})")
    for s, om, os_, mm, ms in zip(strengths, original_means, original_stds, modified_means, modified_stds):
        print(
            f"  strength {s:>3}: Original = {om:.4f} (+/- {os_:.4f}), "
            f"Modified = {mm:.4f} (+/- {ms:.4f})"
        )
    print("Saved plot2_single_attacker_strength_sweep.pdf / .png\n")


def plot_multi_attacker_contribution(histories):
    """
    Plot 3: sum of self-contributions of all malicious clients,
    over rounds, Original vs Modified, averaged over seeds.

    Both fixed attacker-count scenarios (2 and 3 attackers, strength
    1.0 each) are plotted together so the figure also shows how the
    aggregated attack effect scales with the number of colluding
    attackers, not just with strength (as in Plot 2).
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    # (scenario, version) -> (color, linestyle, label)
    style = {
        ("multi_2", "original"): ("C3", "--", "Original, 2 attackers"),
        ("multi_2", "modified"): ("C0", "--", "Modified, 2 attackers"),
        ("multi_3", "original"): ("C3", "-",  "Original, 3 attackers"),
        ("multi_3", "modified"): ("C0", "-",  "Modified, 3 attackers"),
    }

    rounds = None

    for scenario_name in MULTI_ATTACKER_SCENARIOS:

        for version in ["original", "modified"]:

            per_seed_series = []

            for h in histories:

                # multi_2 uses the first 2 ids of the fixed set,
                # multi_3 uses all 3 - read whatever ids were
                # actually used for that scenario from history if
                # available, else fall back to multi_attacker_ids.
                attacker_ids = h.get(
                    f"{scenario_name}_attacker_ids",
                    h.get("multi_attacker_ids", []),
                )

                if scenario_name == "multi_2":
                    attacker_ids = attacker_ids[:2]

                series = self_contribution_per_round(
                    h, scenario_name, version, attacker_ids
                )

                per_seed_series.append(series)

            stacked = np.stack(per_seed_series)
            mean_series = np.mean(stacked, axis=0)
            std_series = np.std(stacked, axis=0)

            if rounds is None:
                rounds = np.arange(1, stacked.shape[1] + 1)

            color, linestyle, label = style[(scenario_name, version)]

            ax.plot(
                rounds, mean_series, marker="o", markersize=3,
                color=color, linestyle=linestyle, label=label,
            )
            ax.fill_between(
                rounds,
                mean_series - std_series,
                mean_series + std_series,
                color=color, alpha=0.12,
            )

            print(
                f"[{scenario_name} / {version}] final round: "
                f"{mean_series[-1]:.4f} (+/- {std_series[-1]:.4f})"
            )

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel(r"Aggregated contribution $A_M(t) = \sum_{i \in M} \phi_i^{(t)}(i)$")
    ax.set_title("Multiple attackers (strength 1.0): aggregated contribution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot3_multi_attacker_contribution.pdf")
    fig.savefig(FIGURES_DIR / "plot3_multi_attacker_contribution.png", dpi=200)
    plt.close(fig)

    print("Saved plot3_multi_attacker_contribution.pdf / .png\n")


def main():

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
    })

    histories = load_all_seeds()

    plot_honest_ranking_correlation(histories)
    plot_single_attacker_strength_sweep(histories)
    plot_multi_attacker_contribution(histories)

    print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()