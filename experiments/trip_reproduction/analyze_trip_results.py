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

    # Zoom into the observed range instead of the full [-1, 1] scale,
    # with a small padding so points don't sit flush on the axes.
    y_min = np.nanmin(mean_rho - std_rho)
    y_max = np.nanmax(mean_rho + std_rho)
    padding = max(0.02, 0.1 * (y_max - y_min))
    ax.set_ylim(
        max(-1.0, y_min - padding),
        min(1.0, y_max + padding),
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot1_honest_ranking_correlation.pdf")
    fig.savefig(FIGURES_DIR / "plot1_honest_ranking_correlation.png", dpi=200)
    plt.close(fig)

    print(f"\nPlot 1 saved. Final-round mean rho: {mean_rho[-1]:.4f} (+/- {std_rho[-1]:.4f})")

    # Overall summary across the whole trajectory: all rounds and
    # all seeds pooled into one distribution, not just the final
    # round. Useful as a single number to cite in text.
    all_values = per_seed_rhos[np.isfinite(per_seed_rhos)]
    overall_mean = float(np.mean(all_values))
    overall_std = float(np.std(all_values))

    print(
        f"Overall (all {per_seed_rhos.shape[1]} rounds x "
        f"{per_seed_rhos.shape[0]} seeds pooled): "
        f"mean rho = {overall_mean:.4f} (+/- {overall_std:.4f})"
    )

    # Round 1 is a known outlier (contribution vectors are still
    # mostly zero-filled at that point - see earlier discussion),
    # so also report the same summary excluding it, for reference.
    if per_seed_rhos.shape[1] > 1:
        excl_r1 = per_seed_rhos[:, 1:]
        excl_r1_values = excl_r1[np.isfinite(excl_r1)]
        excl_mean = float(np.mean(excl_r1_values))
        excl_std = float(np.std(excl_r1_values))
        print(
            f"Overall excluding round 1: "
            f"mean rho = {excl_mean:.4f} (+/- {excl_std:.4f})"
        )


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

    # Modified is invariant to strength by construction (the
    # self-entry is always zeroed regardless of the reported fake
    # value), so instead of plotting near-identical points per
    # strength, pool all (strength, seed) observations into one
    # overall estimate and draw it as a flat line - same treatment
    # as the honest baseline. std is kept only for the console
    # summary below, not plotted (no shaded bands / error bars).
    modified_pooled = np.array(
        [v for s in strengths for v in modified_vals[s]]
    )
    modified_mean = float(np.mean(modified_pooled))
    modified_std = float(np.std(modified_pooled))

    x = np.arange(len(strengths))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(
        honest_mean, color="gray", linestyle="--",
        label="Honest baseline (no attack, Original)",
    )

    ax.axhline(
        modified_mean, color="C0", linestyle="-",
        label="Modified TRIP-Shapley (all strengths)",
    )

    ax.plot(
        x, original_means,
        marker="o", color="C3",
        label="Original TRIP-Shapley",
    )

    # Value labels so approximate magnitudes are readable directly
    # from the figure, without having to decode a log axis by eye.
    for xi, mean in zip(x, original_means):
        ax.annotate(
            f"{mean:.1f}",
            xy=(xi, mean),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color="C3",
        )

    # Both Modified and the honest baseline are flat lines now, so
    # each gets a single label rather than one per x position. They
    # sit close together in value, so they're offset in opposite
    # vertical directions from their respective lines to avoid
    # overlapping each other.
    ax.annotate(
        f"Modified ≈ {modified_mean:.3f}",
        xy=(x[-1], modified_mean),
        xytext=(6, -10),
        textcoords="offset points",
        ha="left", va="top",
        fontsize=7.5, color="C0",
    )

    ax.annotate(
        f"honest ≈ {honest_mean:.3f}",
        xy=(x[-1], honest_mean),
        xytext=(6, 10),
        textcoords="offset points",
        ha="left", va="bottom",
        fontsize=7.5, color="gray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in strengths])
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.9)
    ax.set_xlabel("Reported fake self-contribution strength")
    ax.set_ylabel(r"Attacker final contribution $\phi_i^{(T)}(i)$ (log scale)")
    ax.set_title("Single attacker: final contribution vs. attack strength")

    # Original spans ~4 to ~220 across strengths, while the honest
    # baseline and Modified sit near 0.02-0.03 - on a linear axis
    # these get squashed flat near zero. Log scale keeps both
    # visible and also makes the (roughly multiplicative) growth
    # of Original with strength read as a straight-ish line.
    all_positive = (
        honest_mean > 0
        and np.all(original_means > 0)
        and modified_mean > 0
    )

    if all_positive:
        ax.set_yscale("log")
    else:
        print(
            "Warning: non-positive values present, keeping linear "
            "y-axis (log scale requires all-positive data)."
        )

    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot2_single_attacker_strength_sweep.pdf")
    fig.savefig(FIGURES_DIR / "plot2_single_attacker_strength_sweep.png", dpi=200)
    plt.close(fig)

    print(f"\nHonest baseline (Original, no attack): {honest_mean:.4f} (+/- {honest_std:.4f})")
    print(f"Modified (pooled across all strengths): {modified_mean:.4f} (+/- {modified_std:.4f})")
    for s, om, os_ in zip(strengths, original_means, original_stds):
        print(f"  strength {s:>3}: Original = {om:.4f} (+/- {os_:.4f})")
    for s in strengths:
        print(
            f"  strength {s:>3}: Modified = "
            f"{np.mean(modified_vals[s]):.4f} (+/- {np.std(modified_vals[s]):.4f}) "
            f"[shown pooled in the plot]"
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
    all_mean_series = []

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

            all_mean_series.append(mean_series)

            ax.plot(
                rounds, mean_series, marker="o", markersize=3,
                color=color, linestyle=linestyle, label=label,
            )

            print(
                f"[{scenario_name} / {version}] final round: "
                f"{mean_series[-1]:.4f} (+/- {std_series[-1]:.4f})"
            )

    ax.set_xlabel("Communication round $t$")
    ax.set_ylabel(
        r"Aggregated contribution $A_M(t) = \sum_{i \in M} \phi_i^{(t)}(i)$"
        "\n(symlog scale)"
    )
    ax.set_title("Multiple attackers (strength 1.0): aggregated contribution")

    # Original (2/3 colluding attackers) reaches ~12, while Modified
    # sits near 0 - and can be exactly 0 at round 1, since phi^(0)=0
    # and the modified self-entry is always zeroed regardless of
    # value, so the propagated term is also 0 that round. A plain
    # log scale can't handle an exact zero; symlog uses a small
    # linear region around 0 and switches to log further out, so
    # both the near-zero Modified curves and the much larger
    # Original curves stay visible on the same axis.
    all_values = np.concatenate(all_mean_series)
    nonzero_abs = np.abs(all_values[np.abs(all_values) > 1e-9])

    linthresh = float(np.min(nonzero_abs)) / 2 if nonzero_abs.size else 0.01
    linthresh = max(linthresh, 1e-4)  # avoid a degenerately tiny linear region

    ax.set_yscale("symlog", linthresh=linthresh)

    ax.grid(True, alpha=0.3, which="both")
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