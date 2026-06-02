import torch
import torch.nn.functional as F

class IncentiveTracker:
    def __init__(self, num_clients, decay=0.9):
        self.num_clients = num_clients
        self.decay = decay

        self.scores = [0.0 for _ in range(num_clients)]
        self.history = []  # per-round scores

    def flatten(self, state_dict):
        return torch.cat([v.flatten().cpu() for v in state_dict.values()])

    def cosine(self, a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    def update(self, client_deltas, global_delta):
        round_scores = []

        g = self.flatten(global_delta)

        for i, delta in enumerate(client_deltas):
            d = self.flatten(delta)

            score = self.cosine(d, g)

            # exponential moving average
            self.scores[i] = (
                self.decay * self.scores[i]
                + (1 - self.decay) * score
            )

            round_scores.append(score)

        self.history.append(round_scores)
        return round_scores

    def ranking(self):
        return sorted(
            enumerate(self.scores),
            key=lambda x: x[1],
            reverse=True
        )