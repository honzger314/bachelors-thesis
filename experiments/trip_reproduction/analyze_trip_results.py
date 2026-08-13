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

TOPOLOGY = "watts_strogatz"
NUM_CLIENTS = 10
ROUNDS = 20


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
        print(f"  contribution snapshots: {len(h['contributions'])}")
        histories.append(h)
    return histories


# ---------------------------------------------------------------------
# Inspection (kept from the original single-file script, unchanged)
# ---------------------------------------------------------------------

def inspect_history(history):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("Keys:", list(history.keys()))

    for key in ["topology", "average_degree", "rewire_prob",
                "num_clients", "rounds"]:
        if key in history:
            print(f"{key}: {history[key]}")

    accuracy = history.get("accuracy", [])
    client_accuracy = history.get("client_accuracy", [])
    contributions = history.get("contributions", [])
    lcvs = history.get("lcv_vectors", [])

    print(f"\nNumber of LCV snapshots: {len(lcvs)}")
    if lcvs:
        print(f"LCV snapshot clients: {sorted(lcvs[-1].keys())}")

    print(f"Number of accuracy entries: {len(accuracy)}")
    print(f"Number of client-accuracy entries: {len(client_accuracy)}")
    print(f"Number of contribution snapshots: {len(contributions)}")

    if accuracy:
        print(f"Accuracy first round: {accuracy[0]:.4f}")
        print(f"Accuracy final round: {accuracy[-1]:.4f}")
        print(f"Accuracy improvement: {accuracy[-1] - accuracy[0]:+.4f}")

    if client_accuracy:
        arr = np.asarray(client_accuracy, dtype=float)
        print(f"Client accuracy history shape: {arr.shape}")
        print(f"Final client accuracies: {np.round(arr[-1], 4)}")

    if contributions:
        final = contributions[-1]  # {"original": {...}, "modified": {...}}

        for version in ["original", "modified"]:
            if version not in final:
                continue

            version_dict = final[version]
            client_ids = sorted(version_dict.keys())

            matrix = np.stack(
                [np.asarray(version_dict[cid], dtype=float) for cid in client_ids]
            )

            finite = np.isfinite(matrix).all()
            diagonal = np.diag(matrix)

            print(f"\n[{version}] matrix shape: {matrix.shape}, finite: {finite}")
            print(f"[{version}] self-contribution diagonal: {np.round(diagonal, 6)}")
            print(f"[{version}] max |self-contribution|: {np.max(np.abs(diagonal)):.6g}")


# ---------------------------------------------------------------------
# Ranking correlation (Plot 1 equivalent, adapted to flat contributions)
# ---------------------------------------------------------------------

def get_round_matrix(history, version, round_idx):
    """
    Returns the contribution matrix for a given version
    ("original" or "modified") and round index (0-based).

    Unlike the ring-topology / attack-scenario branch, this branch's
    history["contributions"] is a flat list (one entry per round),
    not nested by scenario name.
    """

    entry = history["contributions"][round_idx][version]
    client_ids = sorted(entry.keys())

    matrix = np.stack(
        [np.asarray(entry[cid], dtype=float) for cid in client_ids]
    )

    return matrix, client_ids


def get_final_matrix(history, version):
    return get_round_matrix(history, version, -1)


def spearman_per_round(history):
    """
    For each round, computes the Spearman correlation between the
    original and modified contribution rows for every receiving
    client, then averages across receivers.

    Returns an array of shape (num_rounds,).
    """

    n_rounds = len(history["contributions"])
    rhos = np.full(n_rounds, np.nan)

    for r in range(n_rounds):

        orig_matrix, client_ids = get_round_matrix(history, "original", r)
        mod_matrix, _ = get_round_matrix(history, "modified", r)

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
    Spearman correlation between original and modified contribution
    rankings, per round, averaged over seeds (mean +/- std band),
    on the Watts-Strogatz topology.
    """

    per_seed_rhos = np.stack(
        [spearman_per_round(h) for h in histories]
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
    ax.set_title(
        "Honest scenario, Watts-Strogatz: ranking correlation\n"
        "(Original vs. Modified TRIP-Shapley)"
    )

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
    fig.savefig(FIGURES_DIR / "plot1_watts_strogatz_ranking_correlation.pdf")
    fig.savefig(FIGURES_DIR / "plot1_watts_strogatz_ranking_correlation.png", dpi=200)
    plt.close(fig)

    print(f"\nPlot 1 (Watts-Strogatz) saved. Final-round mean rho: {mean_rho[-1]:.4f} (+/- {std_rho[-1]:.4f})")

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
    # mostly zero-filled at that point), so also report the same
    # summary excluding it, for reference.
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
# Final-round diagnostics (kept from the original single-file script)
# ---------------------------------------------------------------------

def plot_final_contributions(history, version):

    matrix, client_ids = get_final_matrix(history, version)

    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Contribution")
    plt.xlabel("Contributor")
    plt.ylabel("Receiving client")
    plt.title(f"Final contribution matrix - {version} (Watts-Strogatz)")
    plt.xticks(range(len(client_ids)), client_ids)
    plt.yticks(range(len(client_ids)), client_ids)
    plt.tight_layout()
    plt.show()


def compare_original_vs_modified(history):

    original_matrix, client_ids = get_final_matrix(history, "original")
    modified_matrix, _ = get_final_matrix(history, "modified")

    print("\n" + "=" * 70)
    print("ORIGINAL vs MODIFIED (Watts-Strogatz, honest, single seed)")
    print("=" * 70)

    print(
        f"Max |self-contribution|, original: "
        f"{np.max(np.abs(np.diag(original_matrix))):.6g}"
    )
    print(
        f"Max |self-contribution|, modified: "
        f"{np.max(np.abs(np.diag(modified_matrix))):.6g}"
    )

    rank_changes = []
    for row, cid in enumerate(client_ids):
        orig_order = np.argsort(-original_matrix[row])
        mod_order = np.argsort(-modified_matrix[row])
        same = np.array_equal(orig_order, mod_order)
        rank_changes.append(not same)

    print(
        f"Receiver rows with changed contributor ranking: "
        f"{sum(rank_changes)}/{len(rank_changes)}"
    )

    diff = modified_matrix - original_matrix
    print(f"Mean absolute contribution difference: {np.mean(np.abs(diff)):.6f}")
    print(f"Max absolute contribution difference: {np.max(np.abs(diff)):.6f}")


def main():

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
    })

    histories = load_all_seeds()

    # Detailed single-seed inspection uses the first seed only.
    inspect_history(histories[0])

    # Multi-seed plot (this branch's equivalent of Plot 1).
    plot_honest_ranking_correlation(histories)

    # Single-seed diagnostics, first seed only.
    plot_final_contributions(histories[0], "original")
    plot_final_contributions(histories[0], "modified")
    compare_original_vs_modified(histories[0])

    print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()