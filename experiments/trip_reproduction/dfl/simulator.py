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

    Computes one (honest) LCV vector per client per round, then
    fans it out to several coordinators, each representing a
    different attack scenario:

        clean       - no attack
        single_s1   - one fixed malicious client, strength 1.0
        single_s5   - one fixed malicious client, strength 5.0
        single_s10  - one fixed malicious client, strength 10.0
        single_s20  - one fixed malicious client, strength 20.0
        single_s50  - one fixed malicious client, strength 50.0
        multi_fixed - a fixed set of malicious clients, strength 1.0

    Every coordinator internally tracks both original and modified
    (self-contribution removed) TRIP-Shapley contributions, as
    implemented by Coordinator.update_round.
    """

    def __init__(
        self,
        num_clients=10,
        rounds=20,
        local_epochs=1,
        batch_size=64,
        topology="ring",
        device="cpu",
        single_attacker_id=0,
        multi_attacker_ids=None,
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device

        self.single_attacker_id = single_attacker_id
        self.multi_attacker_ids = multi_attacker_ids or []

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
        # Network (topology preserved)
        #

        self.network = Network(
            num_clients=num_clients,
            topology=topology
        )

        #
        # Attack scenarios: name -> (malicious_ids, strength)
        # strength=None means no corruption applied (clean)
        #

        self.scenarios = {
            "clean":       (set(), None),
            "single_s1":   ({self.single_attacker_id}, 1.0),
            "single_s5":   ({self.single_attacker_id}, 5.0),
            "single_s10":  ({self.single_attacker_id}, 10.0),
            "single_s20":  ({self.single_attacker_id}, 20.0),
            "single_s50":  ({self.single_attacker_id}, 50.0),
            "multi_2":  (set(multi_attacker_ids[:2]), 1.0),   # first 2 of the fixed set
            "multi_3":  (set(multi_attacker_ids), 1.0),        # all 3 (rename from multi_fixed)
        }

        #
        # One coordinator per scenario, each handling both
        # original and modified versions internally
        #

        self.coordinators = {
            name: Coordinator(num_clients=num_clients)
            for name in self.scenarios
        }

        #
        # History
        #

        self.history = {

            "accuracy": [],
            "client_accuracy": [],

            "lcv_vectors": [],

            "contributions": {
                name: []
                for name in self.scenarios
            },

            "topology": topology,
            "num_clients": num_clients,
            "rounds": rounds,

            "single_attacker_id": self.single_attacker_id,
            "multi_attacker_ids": self.multi_attacker_ids,
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
        # 3. Compute LCV once (honest, no attack applied here)
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
        # 4. Update every scenario's coordinator
        #
        # Each scenario clones the honest lcv_dict and, if
        # applicable, overwrites the self-entry of malicious
        # clients with the scenario's fake strength value
        # before propagation. Coordinator.update_round then
        # handles the original/modified split as before.
        #

        print(
            "\n--- Updating coordinators ---"
        )

        for name, (malicious_ids, strength) in self.scenarios.items():

            scenario_lcv_dict = {}

            for cid, vec in lcv_dict.items():

                v = vec.clone()

                if strength is not None and cid in malicious_ids:
                    v[cid] = strength

                scenario_lcv_dict[cid] = v

            self.coordinators[name].update_round(
                scenario_lcv_dict,
                self.network
            )

            self.history["contributions"][name].append(
                copy.deepcopy(
                    self.coordinators[name].get_all_contributions()
                )
            )

        print(
            "Coordinator updates complete"
        )

        #
        # 5. Aggregate models (attack-agnostic)
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