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
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

        return model.state_dict()