import os
import json
import gc

from main import run_experiment

# ------------------------
# GRID SETTINGS
# ------------------------

SEEDS = [1, 2, 3]
ALPHAS = [0.1, 1.0, 10.0]

PARTICIPATION_RATES = [0.25, 0.5, 0.75]

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

    total_runs = (
        len(SEEDS)
        * len(ALPHAS)
        * len(PARTICIPATION_RATES)
    )

    run_id = 0

    for seed in SEEDS:
        for alpha in ALPHAS:
            for participation in PARTICIPATION_RATES:

                run_id += 1

                run_name = (
                    f"run_{run_id}"
                    f"_seed_{seed}"
                    f"_alpha_{alpha}"
                    f"_participation_{participation}"
                )

                out_dir = os.path.join(ROOT, run_name)
                os.makedirs(out_dir, exist_ok=True)

                print("\n" + "=" * 70)
                print(f"RUN {run_id}/{total_runs}")
                print(
                    f"seed={seed}, "
                    f"alpha={alpha}, "
                    f"participation={participation}"
                )
                print("=" * 70)

                # ------------------------
                # RUN EXPERIMENT
                # ------------------------

                result, df = run_experiment(
                    agents=AGENTS,
                    alpha=alpha,
                    rounds=ROUNDS,
                    seed=seed,
                    participation_rate=participation,
                    return_history=True
                )

                # ------------------------
                # SAVE SUMMARY
                # ------------------------

                with open(
                    os.path.join(out_dir, "summary.json"),
                    "w"
                ) as f:
                    json.dump(result, f, indent=2)

                # ------------------------
                # SAVE TELEMETRY
                # ------------------------

                df.to_parquet(
                    os.path.join(out_dir, "telemetry.parquet")
                )

                del df
                gc.collect()

    print("\nALL EXPERIMENTS DONE")


if __name__ == "__main__":
    main()