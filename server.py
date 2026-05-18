import copy
import torch
from itertools import combinations
from model import CNN

class Server:
    def __init__(self, clients):
        self.global_model = CNN()
        self.clients = clients

    def set_weights(self, weights):
        self.global_model.load_state_dict(weights)

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

    # -------------------------
    # FULL SHAPLEY AGGREGATION
    # -------------------------
    def shapley_aggregate(self, client_updates, eval_fn=None):
        """
        DEBUG MODE:
        - No evaluation (fast execution)
        - Shapley structure preserved
        - Used only to verify correctness of pipeline
        """

        n = len(client_updates)
        base_model = copy.deepcopy(self.global_model.state_dict())

        contributions = [0.0 for _ in range(n)]
        clients = list(range(n))

        # collect subsets once (for progress tracking)
        subsets = []
        for r in range(n + 1):
            subsets.extend(list(combinations(clients, r)))

        total_subsets = len(subsets)

        # iterate over all subsets
        for idx, subset in enumerate(subsets):

            # progress print
            if idx % 2 == 0 or idx == total_subsets - 1:
                print(f"[Shapley] Subset {idx+1}/{total_subsets}")

            # build model for subset
            model = CNN()
            model.load_state_dict(base_model)

            # apply updates in subset (sequential overwrite)
            for i in subset:
                model.load_state_dict(client_updates[i])

            # -------------------------
            # DEBUG SCORE (NO EVAL)
            # -------------------------
            score = 0.0

            # marginal contributions
            for i in clients:
                if i not in subset:

                    subset_with_i = tuple(sorted(subset + (i,)))

                    model_i = CNN()
                    model_i.load_state_dict(base_model)

                    for j in subset_with_i:
                        model_i.load_state_dict(client_updates[j])

                    # DEBUG SCORE (NO EVAL)
                    score_i = 0.0

                    contributions[i] += score_i - score

        # normalize Shapley values safely
        total = sum(contributions)

        if total == 0:
            weights = [1.0 / n for _ in range(n)]
        else:
            weights = [c / total for c in contributions]

        # weighted aggregation
        new_state = copy.deepcopy(base_model)

        for key in new_state.keys():
            new_state[key] = sum(
                weights[i] * client_updates[i][key] for i in range(n)
            )

        return new_state, weights