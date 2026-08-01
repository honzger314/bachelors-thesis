import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
ORIGINAL_FILE = RESULTS_DIR / "ring_10clients_10rounds_original_lcv_seed42_malicious1.pkl"
MODIFIED_FILE = RESULTS_DIR / "ring_10clients_10rounds_modified_lcv_seed42_malicious1.pkl"

def load_result(path):
    print(f"\nLoading: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def inspect_history(name, history):
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    print("Keys:", list(history.keys()))

    for key in ["topology", "num_clients", "rounds", "lcv_method"]:
        if key in history:
            print(f"{key}: {history[key]}")

    accuracy = history.get("accuracy", [])
    client_accuracy = history.get("client_accuracy", [])
    contributions = history.get("contributions", [])
    lcvs = history.get("lcv_vectors", [])

    print(f"Number of LCV snapshots: {len(lcvs)}")

    if lcvs:
        print(
            f"LCV snapshot clients: "
            f"{sorted(lcvs[-1].keys())}"
        )

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
        # contributions is expected to be:
        # list[round][client_id] -> vector
        n_rounds = len(contributions)
        client_ids = sorted(contributions[-1].keys())
        n_clients = len(client_ids)

        print(f"Contribution snapshots: {n_rounds}")
        print(f"Number of clients in final snapshot: {n_clients}")

        final_matrix = np.stack(
            [np.asarray(contributions[-1][cid], dtype=float) for cid in client_ids]
        )
        print(f"Final contribution matrix shape: {final_matrix.shape}")

        # Basic sanity checks
        finite = np.isfinite(final_matrix).all()
        print(f"Final contribution matrix finite: {finite}")

        # Row = receiver, column = contributor
        diagonal = np.diag(final_matrix)
        print(f"Final diagonal (self-contribution entries): {np.round(diagonal, 6)}")
        print(f"Max absolute self-contribution: {np.max(np.abs(diagonal)):.6g}")

        # Show largest contributor for each receiver
        for row_idx, cid in enumerate(client_ids):
            row = final_matrix[row_idx].copy()
            top = int(np.argmax(row))
            print(
                f"  Client {cid}: top contributor = {top}, "
                f"value = {row[top]:.6f}"
            )


def plot_accuracy(original, modified):
    plt.figure()
    plt.plot(
        np.arange(1, len(original["accuracy"]) + 1),
        original["accuracy"],
        marker="o",
        label="Original LCV",
    )
    plt.plot(
        np.arange(1, len(modified["accuracy"]) + 1),
        modified["accuracy"],
        marker="o",
        label="Modified LCV",
    )
    plt.xlabel("Round")
    plt.ylabel("Mean test accuracy")
    plt.title("Mean accuracy over rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_final_contributions(history, title):

    if not history.get("contributions"):
        print("No contribution data available")
        return

    contributions = history["contributions"][-1]
    client_ids = sorted(contributions.keys())

    matrix = np.stack(
        [np.asarray(contributions[cid], dtype=float) for cid in client_ids]
    )

    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Contribution")
    plt.xlabel("Contributor")
    plt.ylabel("Receiving client")
    plt.title(title)
    plt.xticks(range(len(client_ids)), client_ids)
    plt.yticks(range(len(client_ids)), client_ids)
    plt.tight_layout()
    plt.show()


def compare_final_contributions(original, modified):
    original_final = original["contributions"][-1]
    modified_final = modified["contributions"][-1]

    client_ids = sorted(original_final.keys())

    original_matrix = np.stack(
        [np.asarray(original_final[cid], dtype=float) for cid in client_ids]
    )
    modified_matrix = np.stack(
        [np.asarray(modified_final[cid], dtype=float) for cid in client_ids]
    )

    print("\n" + "=" * 70)
    print("DIRECT COMPARISON")
    print("=" * 70)

    print(
        f"Final mean accuracy - original: "
        f"{original['accuracy'][-1]:.4f}"
    )
    print(
        f"Final mean accuracy - modified: "
        f"{modified['accuracy'][-1]:.4f}"
    )
    print(
        f"Difference (modified - original): "
        f"{modified['accuracy'][-1] - original['accuracy'][-1]:+.4f}"
    )

    print(
        f"\nMax absolute self-contribution, original: "
        f"{np.max(np.abs(np.diag(original_matrix))):.6g}"
    )
    print(
        f"Max absolute self-contribution, modified: "
        f"{np.max(np.abs(np.diag(modified_matrix))):.6g}"
    )

    # Compare contributor rankings for each receiver
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

    # Matrix-level summary
    diff = modified_matrix - original_matrix
    print(
        f"Mean absolute contribution difference: "
        f"{np.mean(np.abs(diff)):.6f}"
    )
    print(
        f"Max absolute contribution difference: "
        f"{np.max(np.abs(diff)):.6f}"
    )


def main():
    original = load_result(ORIGINAL_FILE)
    modified = load_result(MODIFIED_FILE)

    inspect_history("ORIGINAL", original)
    inspect_history("MODIFIED", modified)

    # Ensure these really are the same experiment apart from LCV method.
    print("\n" + "=" * 70)
    print("EXPERIMENT CONSISTENCY CHECK")
    print("=" * 70)

    for key in ["topology", "num_clients", "rounds"]:
        a = original.get(key)
        b = modified.get(key)
        print(f"{key}: {'OK' if a == b else f'MISMATCH ({a} vs {b})'}")

    plot_accuracy(original, modified)

    plot_final_contributions(
        original,
        "Final contribution matrix - Original LCV"
    )

    plot_final_contributions(
        modified,
        "Final contribution matrix - Modified LCV"
    )

    compare_final_contributions(original, modified)


if __name__ == "__main__":
    main()