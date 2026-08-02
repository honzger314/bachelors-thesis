import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")

# One file now contains every scenario (clean, single_s1..s50, multi_fixed)
# and both original/modified contribution tracking within each scenario.
RESULTS_FILE = RESULTS_DIR / "CIFAR10_ring_10clients_20rounds_seed_1.pkl"


def load_result(path):
    print(f"\nLoading: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def inspect_history(history):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("Keys:", list(history.keys()))

    for key in ["topology", "num_clients", "rounds",
                "single_attacker_id", "multi_attacker_ids"]:
        if key in history:
            print(f"{key}: {history[key]}")

    accuracy = history.get("accuracy", [])
    client_accuracy = history.get("client_accuracy", [])
    lcvs = history.get("lcv_vectors", [])
    contributions = history.get("contributions", {})

    print(f"\nNumber of LCV snapshots: {len(lcvs)}")
    if lcvs:
        print(f"LCV snapshot clients: {sorted(lcvs[-1].keys())}")

    print(f"Number of accuracy entries: {len(accuracy)}")
    print(f"Number of client-accuracy entries: {len(client_accuracy)}")

    if accuracy:
        print(f"Accuracy first round: {accuracy[0]:.4f}")
        print(f"Accuracy final round: {accuracy[-1]:.4f}")
        print(f"Accuracy improvement: {accuracy[-1] - accuracy[0]:+.4f}")

    if client_accuracy:
        arr = np.asarray(client_accuracy, dtype=float)
        print(f"Client accuracy history shape: {arr.shape}")
        print(f"Final client accuracies: {np.round(arr[-1], 4)}")

    print(f"\nScenarios found: {list(contributions.keys())}")

    for scenario_name, rounds_list in contributions.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"  Rounds recorded: {len(rounds_list)}")

        if not rounds_list:
            continue

        final = rounds_list[-1]  # {"original": {...}, "modified": {...}}

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

            print(f"  [{version}] matrix shape: {matrix.shape}, finite: {finite}")
            print(f"  [{version}] self-contribution diagonal: {np.round(diagonal, 6)}")
            print(f"  [{version}] max |self-contribution|: {np.max(np.abs(diagonal)):.6g}")


def get_final_matrix(history, scenario_name, version):
    """
    Returns the final-round contribution matrix for a given
    scenario ("clean", "single_s1", ..., "multi_fixed") and
    version ("original" or "modified").
    """

    contributions = history["contributions"][scenario_name]
    final = contributions[-1][version]
    client_ids = sorted(final.keys())

    matrix = np.stack(
        [np.asarray(final[cid], dtype=float) for cid in client_ids]
    )

    return matrix, client_ids


def plot_accuracy(history):
    """
    Single accuracy curve. Attack scenarios only affect diagnostic
    contribution tracking, not aggregation, so there is only one
    accuracy trajectory for the whole experiment.
    """

    accuracy = history["accuracy"]

    plt.figure()
    plt.plot(
        np.arange(1, len(accuracy) + 1),
        accuracy,
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Mean test accuracy")
    plt.title("Mean accuracy over rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_final_contributions(history, scenario_name, version):

    matrix, client_ids = get_final_matrix(history, scenario_name, version)

    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Contribution")
    plt.xlabel("Contributor")
    plt.ylabel("Receiving client")
    plt.title(f"Final contributions - {scenario_name} ({version})")
    plt.xticks(range(len(client_ids)), client_ids)
    plt.yticks(range(len(client_ids)), client_ids)
    plt.tight_layout()
    plt.show()


def compare_original_vs_modified(history, scenario_name):
    """
    Within one scenario, compares original vs modified contribution
    tracking: does modified successfully suppress self-contribution
    inflation while original does not.
    """

    original_matrix, client_ids = get_final_matrix(history, scenario_name, "original")
    modified_matrix, _ = get_final_matrix(history, scenario_name, "modified")

    print("\n" + "=" * 70)
    print(f"ORIGINAL vs MODIFIED — scenario: {scenario_name}")
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


def compare_attack_strengths(history, attacker_id, version="original"):
    """
    Across the single_s1/s5/s10/s20/s50 scenarios, shows how the
    fixed attacker's self-contribution grows with strength, under
    a given version (original or modified).
    """

    strengths = [1, 5, 10, 20, 50]
    scenario_names = [f"single_s{s}" for s in strengths]

    self_contribs = []

    for name in scenario_names:
        matrix, client_ids = get_final_matrix(history, name, version)
        row_idx = client_ids.index(attacker_id)
        self_contribs.append(matrix[row_idx, row_idx])

    print("\n" + "=" * 70)
    print(f"ATTACK STRENGTH SWEEP — client {attacker_id}, version: {version}")
    print("=" * 70)

    for s, val in zip(strengths, self_contribs):
        print(f"  strength {s:>3}: self-contribution = {val:.6f}")

    plt.figure()
    plt.plot(strengths, self_contribs, marker="o")
    plt.xlabel("Fake LCV strength")
    plt.ylabel("Final self-contribution")
    plt.title(f"Self-contribution vs attack strength ({version})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    history = load_result(RESULTS_FILE)

    inspect_history(history)

    plot_accuracy(history)

    scenario_names = list(history["contributions"].keys())

    for scenario_name in scenario_names:
        for version in ["original", "modified"]:
            plot_final_contributions(history, scenario_name, version)

        compare_original_vs_modified(history, scenario_name)

    attacker_id = history.get("single_attacker_id", 0)

    compare_attack_strengths(history, attacker_id, version="original")
    compare_attack_strengths(history, attacker_id, version="modified")


if __name__ == "__main__":
    main()