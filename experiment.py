import os
import json
from datetime import datetime

from main import run_experiment


# ------------------------
# GRID SETTINGS
# ------------------------
SEEDS = [1]
#SEEDS = [1, 2, 3, 4, 5]

ALPHAS = [0.1]
#ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]

AGENTS = 10
ROUNDS = 20


# ------------------------
# ROOT FOLDER
# ------------------------
ROOT = "results"
os.makedirs(ROOT, exist_ok=True)


# ------------------------
# RUN GRID SEARCH
# ------------------------
def main():

    total_runs = len(SEEDS) * len(ALPHAS)
    run_id = 0

    for seed in SEEDS:
        for alpha in ALPHAS:

            run_id += 1

            run_name = f"run_{run_id}_seed_{seed}_alpha_{alpha}"
            out_dir = os.path.join(ROOT, run_name)
            os.makedirs(out_dir, exist_ok=True)

            print("\n" + "=" * 70)
            print(f"RUN {run_id}/{total_runs}")
            print(f"seed={seed}, alpha={alpha}")
            print("=" * 70)

            result = run_experiment(
                agents=AGENTS,
                alpha=alpha,
                rounds=ROUNDS,
                seed=seed,
                return_history=True
            )

            # ------------------------
            # SAVE RESULTS
            # ------------------------
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(result, f, indent=2)

    print("\nALL EXPERIMENTS DONE")


if __name__ == "__main__":
    main()