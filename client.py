import torch
import copy
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, dataset, model_fn, device, batch_size=32, lr=0.01):
        self.id = client_id
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model_fn = model_fn
        self.lr = lr
        self.device = device

    def train(self, global_model, epochs=1):
        import torch
        import copy

        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        initial_params = {k: v.detach().clone() for k, v in model.state_dict().items()}

        last_grads = None  # we will store last batch gradients

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()

                # capture gradients (flattened once per batch)
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().clone().cpu())

                last_grads = torch.cat([g.flatten() for g in grads])

                optimizer.step()

        final_params = model.state_dict()

        # compute update (FedAvg-style delta)
        update_vec = torch.cat([
            (final_params[k] - initial_params[k]).flatten().cpu()
            for k in final_params
        ])

        return final_params, update_vec, last_grads