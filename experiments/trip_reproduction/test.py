import pickle

filepath = "results4/CIFAR10_ring_10clients_20rounds_seed1.pkl"

with open(filepath, "rb") as f:
    history = pickle.load(f)

for key in sorted(history["contributions"].keys()):
    print(key)