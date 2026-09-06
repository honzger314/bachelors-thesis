import pickle
from collections import Counter, defaultdict

PATH = r"results4\CIFAR10_ring_10clients_20rounds_seed1.pkl"

with open(PATH, "rb") as f:
    data = pickle.load(f)

print("=" * 80)
print("OUTLIER DETECTION AT THRESHOLD = 0.01")
print("=" * 80)

# ------------------------------------------------------------
# Helper to inspect one configuration
# ------------------------------------------------------------

def inspect_config(name):
    print(f"\n{'=' * 80}")
    print(name)
    print("=" * 80)

    oh = data["outlier_history"].get(name)

    if oh is None:
        print("CONFIG NOT FOUND")
        return

    print("Type:", type(oh))
    print("Length:", len(oh))

    if isinstance(oh, dict):
        print("Keys:", list(oh.keys())[:20])

        for k, v in oh.items():
            print(f"\nKey {k!r}:")
            print("  type:", type(v))
            print("  value:", v)

    elif isinstance(oh, list):
        for i, v in enumerate(oh):
            print(f"\nRound {i}:")
            print(" ", v)


# ------------------------------------------------------------
# Find relevant configurations
# ------------------------------------------------------------

configs = [
    "clean__p_0.0__threshold_0.01",
    "stealth_half__p_0.0__threshold_0.01",
    "stealth_full__p_0.0__threshold_0.01",
]

for c in configs:
    inspect_config(c)


# ------------------------------------------------------------
# Also inspect summary metrics for clean @ 0.01
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY METRICS — CLEAN @ 0.01")
print("=" * 80)

name = "clean__p_0.0__threshold_0.01"
metrics = data["summary_metrics"].get(name)

if metrics:
    for k, v in metrics.items():
        print(f"{k}: {v}")


# ------------------------------------------------------------
# Inspect audit structure
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("AUDIT LOG STRUCTURE")
print("=" * 80)

audit = data["audit_logs"].get(name)

print("Type:", type(audit))
print("Length:", len(audit) if audit is not None else None)

if isinstance(audit, dict):
    print("Keys:", list(audit.keys())[:20])

    for k, v in list(audit.items())[:3]:
        print(f"\n{k}:")
        print(v)
elif isinstance(audit, list):
    for i, v in enumerate(audit[:5]):
        print(f"Round {i}: {v}")