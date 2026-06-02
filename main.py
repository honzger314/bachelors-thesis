import torch
from torchvision import datasets, transforms
from client import Client
from server import Server
from model import CNN

# ------------------------
# DATA (IID CIFAR-10)
# ------------------------
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    return train, test

agents = 5

def iid_split(dataset, num_clients=agents):
    shards = [[] for _ in range(num_clients)]

    for i, (x, y) in enumerate(dataset):
        shards[i % num_clients].append((x, y))

    return shards


# ------------------------
# EVAL FUNCTION (for Shapley)
# ------------------------
def make_eval(test_loader):
    def eval_fn(model):
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total

    return eval_fn


# ------------------------
# TRAIN LOOP
# ------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train, test = load_data()

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False
    )

    # move evaluation model to device via closure
    def make_eval(test_loader):
        def eval_fn(model):
            model.to(device)
            model.eval()

            correct, total = 0, 0

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    out = model(x)
                    pred = out.argmax(dim=1)

                    correct += (pred == y).sum().item()
                    total += y.size(0)

            return correct / total

        return eval_fn

    eval_fn = make_eval(test_loader)

    shards = iid_split(train, num_clients=agents)

    clients = [
        Client(i, shards[i], CNN)
        for i in range(agents)
    ]

    server = Server(clients, device)

    global_model = CNN().to(device)

    rounds = 5

    for r in range(rounds):
        print(f"\n--- Round {r} ---")

        client_updates = []

        for c in clients:
            update = c.train(global_model)
            client_updates.append(update)

        new_weights, fedavg_weights = server.fedavg(client_updates)

        global_model.load_state_dict(new_weights)
        server.set_weights(new_weights)

        acc = server.evaluate(test_loader)

        print("Accuracy:", acc)
        print("Weights:", fedavg_weights)


if __name__ == "__main__":
    main()