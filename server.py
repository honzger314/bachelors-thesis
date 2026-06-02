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

    def evaluate(self, test_loader):
        self.global_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                out = self.global_model(x)
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