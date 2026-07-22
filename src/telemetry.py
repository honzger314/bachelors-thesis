import torch
import torch.nn.functional as F
import pandas as pd


class TelemetryCollector:
    def __init__(self):
        self.data = []

        self.prev_global = None
        self.prev_client_updates = None
        self.prev_client_grads = None

    # --------------------------
    # flatten helper (state dict OR tensor)
    # --------------------------
    def flatten(self, obj):
        if isinstance(obj, dict):
            return torch.cat([v.flatten().cpu() for v in obj.values()])
        return obj.flatten().cpu()

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
        client_grads,
        client_losses_before,
        client_losses_after,
        test_loss,
        test_acc,
        loo_scores=None
    ):

        global_vec = self.flatten(global_model.state_dict())

        global_delta = None
        if self.prev_global is not None:
            global_delta = global_vec - self.prev_global

        round_data = []

        for i, update in enumerate(client_updates):

            update_vec = self.flatten(update)

            if client_grads is not None:
                grad_vec = self.flatten(client_grads[i])
            else:
                grad_vec = torch.zeros_like(update_vec)

            record = {
                "client_id": i,
                "round": round_id,

                # Store the full gradient vector
                "gradient": grad_vec.clone(),

                # Core signals
                "update_norm": torch.norm(update_vec).item(),
                "grad_norm": torch.norm(grad_vec).item(),

                "local_loss_before": (
                    client_losses_before[i]
                    if client_losses_before is not None
                    else None
                ),
                "local_loss_after": (
                    client_losses_after[i]
                    if client_losses_after is not None
                    else None
                ),

                # Alignment
                "cosine_to_global": None,
                "cosine_to_prev": None,
                "cosine_grad_update": self.cosine(grad_vec, update_vec),

                # Metadata
                "test_loss": test_loss,
                "test_acc": test_acc,
                "num_samples": None,
            }

            if global_delta is not None:
                record["cosine_to_global"] = self.cosine(update_vec, global_delta)

            if self.prev_client_updates is not None:
                prev = self.flatten(self.prev_client_updates[i])
                record["cosine_to_prev"] = self.cosine(update_vec, prev)

            if loo_scores is not None:
                record["loo_score"] = loo_scores[i]

            round_data.append(record)

        self.data.append(round_data)

        # Update memory
        self.prev_global = global_vec
        self.prev_client_updates = client_updates
        self.prev_client_grads = client_grads

    # --------------------------
    # export dataset
    # --------------------------
    def export(self):
        rows = []

        for round_data in self.data:
            for r in round_data:
                rows.append({
                    "round": r["round"],
                    "client_id": r["client_id"],
                    "update_norm": r["update_norm"],
                    "grad_norm": r["grad_norm"],
                    "cosine_to_global": r["cosine_to_global"],
                    "cosine_to_prev": r["cosine_to_prev"],
                    "cosine_grad_update": r["cosine_grad_update"],
                    "test_acc": r["test_acc"],
                    "test_loss": r["test_loss"],
                    "loo_score": r.get("loo_score", None),
                    # IMPORTANT: flatten gradient for table storage
                    "gradient": r["gradient"].cpu().detach().numpy().tolist(),
                })

        return pd.DataFrame(rows)