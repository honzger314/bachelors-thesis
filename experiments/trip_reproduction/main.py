import torch

from dfl.simulator import DFLSimulator



def main():

    # Automatically use GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )


    simulator = DFLSimulator(
        num_clients=5,
        rounds=3,
        local_epochs=1,
        batch_size=64,
        topology="ring",
        device=device
    )


    clients = simulator.train()


    print("\nFinal client accuracies:")


    accuracies = simulator.evaluate()


    for i, acc in enumerate(accuracies):

        print(
            f"Client {i}: {acc:.4f}"
        )



if __name__ == "__main__":
    main()