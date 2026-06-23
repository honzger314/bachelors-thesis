import copy
import torch
from model import CNN

class Server:
    def __init__(self, clients, device):
        self.device = device
        self.global_model = CNN().to(device)
        self.clients = clients

    def set_weights(self, weights):
        self.global_model.load_state_dict(weights)
        self.global_model.to(self.device)

    def evaluate(self, test_loader, device):
        self.global_model.to(device)
        self.global_model.eval()

        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = self.global_model(x)
                pred = out.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total

    def evaluate_model(self, model, test_loader, device):
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
    
    def fedavg(self, client_updates):
        n = len(client_updates)

        weights = [1.0 / n for _ in range(n)]

        # create empty copy of model structure (safer than copying weights)
        new_state = copy.deepcopy(self.global_model.state_dict())

        for key in new_state.keys():
            new_state[key] = sum(
                weights[i] * client_updates[i][key]
                for i in range(n)
            )

        return new_state, weights
    
    def loo_evaluate(self, client_updates, test_fn):
        """
        Returns per-client LOO contribution scores.
        """

        n = len(client_updates)

        full_state, _ = self.fedavg(client_updates)

        full_model = CNN().to(self.device)
        full_model.load_state_dict(full_state)

        full_acc = test_fn(full_model)

        loo_scores = []

        for i in range(n):

            subset = [
                client_updates[j]
                for j in range(n)
                if j != i
            ]

            loo_state, _ = self.fedavg(subset)

            loo_model = CNN().to(self.device)
            loo_model.load_state_dict(loo_state)

            loo_acc = test_fn(loo_model)

            # marginal contribution
            contribution = full_acc - loo_acc
            loo_scores.append(contribution)

        return full_acc, loo_scores