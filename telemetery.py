import torch
import torch.nn.functional as F
import copy

class TelemetryCollector:
    def __init__(self):
        self.data = []

        self.prev_global = None
        self.prev_client_updates = None

    # --------------------------
    # flatten helper
    # --------------------------
    def flatten(self, state_dict):
        return torch.cat([v.flatten().cpu() for v in state_dict.values()])

    # --------------------------
    # cosine similarity
    # --------------------------
    def cosine(self, a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    # --------------------------
    # main entry per round
    # --------------------------
    def log_round(
        self,
        round_id,
        global_model,
        client_updates,
        client_losses_before,
        client_losses_after,
        test_loss,
        test_acc,
        loo_scores=None
    ):

        global_vec = self.flatten(global_model.state_dict())

        # compute global delta if possible
        global_delta = None
        if self.prev_global is not None:
            global_delta = global_vec - self.prev_global

        round_data = []

        for i, update in enumerate(client_updates):

            update_vec = self.flatten(update)

            record = {
                "client_id": i,
                "round": round_id,

                # ---------------- core signals
                "update_norm": torch.norm(update_vec).item(),
                "local_loss_before": client_losses_before[i],
                "local_loss_after": client_losses_after[i],

                # ---------------- alignment
                "cosine_to_global": None,
                "cosine_to_prev": None,

                # ---------------- metadata placeholders
                "test_loss": test_loss,
                "test_acc": test_acc,
                "num_samples": None,
            }

            # cosine to global update
            if global_delta is not None:
                record["cosine_to_global"] = self.cosine(update_vec, global_delta)

            # temporal stability
            if self.prev_client_updates is not None:
                prev = self.flatten(self.prev_client_updates[i])
                record["cosine_to_prev"] = self.cosine(update_vec, prev)

            # LOO label
            if loo_scores is not None:
                record["loo_score"] = loo_scores[i]

            round_data.append(record)

        self.data.append(round_data)

        # update memory
        self.prev_global = global_vec
        self.prev_client_updates = client_updates

    # --------------------------
    # export dataset
    # --------------------------
    def export(self):
        flat = []
        for round_data in self.data:
            flat.extend(round_data)
        return flat