import copy

import torch
import torch.nn as nn

from models.cnn import create_model
from utils.model_utils import average_models

from trip.lcv import compute_lcv

class Client:
    """
    One decentralized FL client.

    During every round the client stores

        theta_t        : model before local training
        theta_t_half   : model after local training

    matching the notation used in the TRIP-Shapley paper.
    """

    def __init__(
        self,
        client_id,
        train_loader,
        device="cpu",
        learning_rate=0.01,
    ):

        self.id = client_id
        self.device = device
        self.train_loader = train_loader

        self.model = create_model().to(device)

        self.learning_rate = learning_rate

        # θ(t)
        self.pre_model = None

        # θ(t+1/2)
        self.post_model = None

        # Filled in later by TRIP-Shapley
        self.local_contribution_vector = None

    ####################################################################
    # Training
    ####################################################################

    def train_local(self, epochs=1):
        """
        Performs local SGD.

        Before training:
            pre_model = θ(t)

        After training:
            post_model = θ(t+1/2)
        """

        # Save θ(t)
        self.pre_model = copy.deepcopy(self.model)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for _ in range(epochs):

            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

        # Save θ(t+1/2)
        self.post_model = copy.deepcopy(self.model)

    ####################################################################
    # Communication
    ####################################################################

    def create_message(self):
        """
        Creates the object sent to neighbors.

        This exactly matches the information exchanged
        in TRIP-Shapley.
        """

        return {
            "client_id": self.id,
            "pre_model": copy.deepcopy(self.pre_model),
            "post_model": copy.deepcopy(self.post_model),
        }

    ####################################################################
    # Aggregation
    ####################################################################

    def aggregate(self, received_messages, weights=None):
        """
        Aggregates all models in `received_messages`, which
        already includes this client's own message (added once
        by the simulator) — do NOT prepend self.post_model again.

        `weights` is a dict {client_id: weight}, keyed the same
        way as `network.get_weights(...)`. If None, uniform.
        """

        models = [msg["post_model"] for msg in received_messages]
        ids = [msg["client_id"] for msg in received_messages]

        if weights is None:
            weight_list = [1.0 / len(models)] * len(models)
        else:
            weight_list = [weights[cid] for cid in ids]

        new_model = average_models(models, weight_list)

        self.model = new_model.to(self.device)

    ####################################################################
    # Accessors
    ####################################################################

    def get_model(self):
        return self.model

    def get_pre_model(self):
        return self.pre_model

    def get_post_model(self):
        return self.post_model

    ####################################################################
    # Placeholder for TRIP-Shapley
    ####################################################################

    def compute_lcv(
        self,
        received_messages,
        test_loader,
    ):

        #
        # TRIP-Shapley:
        # N(i,t) = neighbors + self
        #
        # `received_messages` already includes this client's
        # own message (the simulator adds it before calling
        # this method), so we must NOT append self again here.
        #

        messages = received_messages


        print(
            f"[Client {self.id}] Starting LCV computation "
            f"with {len(messages)} participants"
        )


        self.local_contribution_vector = compute_lcv(
            messages=messages,
            test_loader=test_loader,
            device=self.device
        )


        print(
            f"[Client {self.id}] Finished LCV computation"
        )


        print(
            f"[Client {self.id}] LCV: "
            f"{self.local_contribution_vector}"
        )


        return self.local_contribution_vector