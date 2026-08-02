import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "CIFAR10_watts_strogatz_10clients_20rounds_seed_1.pkl"


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
        # contributions[-1] -> {"original": {cid: vec}, "modified": {cid: vec}}
        final = contributions[-1]

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

            for row_idx, cid in enumerate(client_ids):
                row = matrix[row_idx].copy()
                top = int(np.argmax(row))
                print(
                    f"  Client {cid}: top contributor = {top}, "
                    f"value = {row[top]:.6f}"
                )


def get_final_matrix(history, version):

    final = history["contributions"][-1][version]
    client_ids = sorted(final.keys())

    matrix = np.stack(
        [np.asarray(final[cid], dtype=float) for cid in client_ids]
    )

    return matrix, client_ids


def plot_accuracy(history):

    accuracy = history["accuracy"]

    plt.figure()
    plt.plot(
        np.arange(1, len(accuracy) + 1),
        accuracy,
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Mean test accuracy")
    plt.title("Mean accuracy over rounds (Watts-Strogatz, honest)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    print("ORIGINAL vs MODIFIED (Watts-Strogatz, honest)")
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
    history = load_result(RESULTS_FILE)

    inspect_history(history)

    plot_accuracy(history)

    plot_final_contributions(history, "original")
    plot_final_contributions(history, "modified")

    compare_original_vs_modified(history)


if __name__ == "__main__":
    main()