import torch
import copy

from dfl.client import Client
from dfl.network import Network
from trip.coordinator import Coordinator
from trip.lcv import compute_lcv
from models.cnn import create_model

from data.dataset import create_client_loaders
from utils.model_utils import evaluate_model


class DFLSimulator:
    """
    Decentralized Federated Learning simulator.

    This branch tests robustness of the TRIP-Shapley method itself
    (original vs modified contribution tracking) on an alternative
    topology (Watts-Strogatz), under the honest scenario only - no
    attackers.

    Computes one (honest) LCV vector per client per round, and
    feeds it into a single coordinator, which internally tracks
    both original and modified (self-contribution removed)
    TRIP-Shapley contributions, as implemented by
    Coordinator.update_round.
    """

    def __init__(
        self,
        num_clients=10,
        rounds=20,
        local_epochs=1,
        batch_size=64,
        topology="watts_strogatz",
        average_degree=4,
        rewire_prob=0.1,
        network_seed=None,
        device="cpu",
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device

        #
        # Dataset
        #

        client_loaders, test_loader = create_client_loaders(
            num_clients=num_clients,
            batch_size=batch_size
        )

        self.test_loader = test_loader

        #
        # LCV function (always the honest Shapley computation)
        #

        lcv_function = compute_lcv

        #
        # Create synchronized clients
        #

        self.clients = []

        global_model = create_model().to(device)

        global_state = copy.deepcopy(
            global_model.state_dict()
        )

        for i in range(num_clients):

            client = Client(
                client_id=i,
                train_loader=client_loaders[i],
                device=device,
                lcv_function=lcv_function,
            )

            client.model.load_state_dict(
                global_state
            )

            self.clients.append(client)

        #
        # Network (Watts-Strogatz on this branch)
        #

        self.network = Network(
            num_clients=num_clients,
            topology=topology,
            average_degree=average_degree,
            rewire_prob=rewire_prob,
            random_seed=network_seed
        )

        #
        # Single coordinator, tracks both original and modified
        # contributions internally.
        #

        self.coordinator = Coordinator(
            num_clients=num_clients
        )

        #
        # History
        #

        self.history = {

            "accuracy": [],
            "client_accuracy": [],

            "lcv_vectors": [],

            "contributions": [],

            "topology": topology,
            "average_degree": average_degree,
            "rewire_prob": rewire_prob,
            "num_clients": num_clients,
            "rounds": rounds,
        }

    def train_round(
        self,
        round_number
    ):

        print("\n======================")
        print(f"Starting round {round_number + 1}")
        print("======================")

        #
        # 1. Local training
        #

        print("\n--- Local training ---")

        for client in self.clients:

            print(
                f"[Round {round_number + 1}] "
                f"Training client {client.id}"
            )

            client.train_local(
                epochs=self.local_epochs
            )

        #
        # 2. Exchange messages
        #

        print("\n--- Creating messages ---")

        messages = {}

        for client in self.clients:

            received = []

            received.append(
                client.create_message()
            )

            for neighbor in self.network.neighbors(
                client.id
            ):

                received.append(
                    self.clients[neighbor].create_message()
                )

            messages[client.id] = received

        #
        # 3. Compute LCV (honest, no attack)
        #

        print(
            "\n--- Computing Local Contribution Vectors ---"
        )

        lcv_dict = {}

        for client in self.clients:

            print(
                f"\n[Client {client.id}] Starting LCV"
            )

            lcv = client.compute_lcv(
                received_messages=messages[client.id],
                test_loader=self.test_loader
            )

            print(
                f"[Client {client.id}] Finished LCV"
            )

            vector = torch.zeros(
                self.num_clients
            )

            for cid, value in lcv.items():

                vector[cid] = value

            lcv_dict[client.id] = vector

        self.history["lcv_vectors"].append(
            copy.deepcopy(lcv_dict)
        )

        #
        # 4. Coordinator update (both original and modified)
        #

        print(
            "\n--- Updating coordinator ---"
        )

        self.coordinator.update_round(
            lcv_dict,
            self.network
        )

        self.history["contributions"].append(
            copy.deepcopy(
                self.coordinator.get_all_contributions()
            )
        )

        print(
            "Coordinator update complete"
        )

        #
        # 5. Aggregate models
        #

        print(
            "\n--- Model aggregation ---"
        )

        for client in self.clients:

            weights = self.network.get_weights(
                client.id
            )

            client.aggregate(
                received_messages=messages[client.id],
                weights=weights
            )

        print(
            f"Round {round_number + 1} complete"
        )

    def evaluate(self):

        accuracies = []

        for client in self.clients:

            acc = evaluate_model(
                client.model,
                self.test_loader,
                self.device
            )

            accuracies.append(acc)

        return accuracies

    def train(self):

        print(
            "Starting DFL training"
        )

        self.network.print_network()

        for r in range(self.rounds):

            self.train_round(r)

            print(
                "\n--- Evaluation ---"
            )

            accuracies = self.evaluate()

            mean_accuracy = sum(
                accuracies
            ) / len(accuracies)

            self.history["client_accuracy"].append(
                accuracies
            )

            self.history["accuracy"].append(
                mean_accuracy
            )

            print(
                f"Round {r+1}/{self.rounds} "
                f"| Mean accuracy: {mean_accuracy:.4f}"
            )

        print(
            "Training finished"
        )

        return self.clients

    def get_history(self):

        return self.history