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

    Computes one ground-truth LCV vector per client per round,
    then creates manipulated reports for different attack scenarios.

    Coordinators track:

        original:
            No defense. Propagates the reported LCV unchanged.

        modified:
            Baseline defense. Removes the client's own contribution
            before propagation.

        audited:
            Probabilistic + outlier-triggered verification.

    IMPORTANT:
        The ground-truth LCV is available to the simulator because
        this is an experimental simulation. In the actual protocol,
        the coordinator would recompute the LCV from the exchanged
        models when an audit is triggered.
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

        # Defense parameters
        audit_probability=0.1,
        audit_threshold=0.001,
        outlier_threshold=0.01,
        seed=42,
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device

        self.single_attacker_id = single_attacker_id
        self.multi_attacker_ids = multi_attacker_ids or []

        # Defense parameters
        self.audit_probability = audit_probability
        self.audit_threshold = audit_threshold
        self.outlier_threshold = outlier_threshold
        self.seed = seed

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------

        client_loaders, test_loader = create_client_loaders(
            num_clients=num_clients,
            batch_size=batch_size
        )

        self.test_loader = test_loader

        # ---------------------------------------------------------
        # LCV function
        # ---------------------------------------------------------

        lcv_function = compute_lcv

        # ---------------------------------------------------------
        # Create synchronized clients
        # ---------------------------------------------------------

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

            client.model.load_state_dict(global_state)

            self.clients.append(client)

        # ---------------------------------------------------------
        # Network
        # ---------------------------------------------------------

        self.network = Network(
            num_clients=num_clients,
            topology=topology
        )

        # ---------------------------------------------------------
        # Attack scenarios
        # ---------------------------------------------------------

        self.scenarios = {

            "clean": (
                set(),
                None
            ),

            "single_s1": (
                {self.single_attacker_id},
                1.0
            ),

            "single_s5": (
                {self.single_attacker_id},
                5.0
            ),

            "single_s10": (
                {self.single_attacker_id},
                10.0
            ),

            "single_s20": (
                {self.single_attacker_id},
                20.0
            ),

            "single_s50": (
                {self.single_attacker_id},
                50.0
            ),

            "multi_2": (
                set(self.multi_attacker_ids[:2]),
                1.0
            ),

            "multi_3": (
                set(self.multi_attacker_ids),
                1.0
            ),
        }

        # ---------------------------------------------------------
        # Coordinators
        # ---------------------------------------------------------
        #
        # Give each coordinator a different seed so that the
        # random audit decisions are independent between scenarios.
        #

        self.coordinators = {}

        for scenario_index, name in enumerate(self.scenarios):

            self.coordinators[name] = Coordinator(
                num_clients=num_clients,
                audit_probability=audit_probability,
                outlier_threshold=outlier_threshold,
                seed=seed,
            )

        # ---------------------------------------------------------
        # History
        # ---------------------------------------------------------

        self.history = {

            "accuracy": [],
            "client_accuracy": [],

            # Ground-truth LCVs
            "lcv_vectors": [],

            # Contributions for each attack scenario
            "contributions": {
                name: []
                for name in self.scenarios
            },

            # Audit information
            "audit_logs": {
                name: []
                for name in self.scenarios
            },

            "topology": topology,
            "num_clients": num_clients,
            "rounds": rounds,

            "single_attacker_id": self.single_attacker_id,
            "multi_attacker_ids": self.multi_attacker_ids,

            "audit_probability": audit_probability,
            "audit_threshold": audit_threshold,
            "outlier_threshold": outlier_threshold,
        }

    def train_round(self, round_number):

        print("\n======================")
        print(f"Starting round {round_number + 1}")
        print("======================")

        # =========================================================
        # 1. Local training
        # =========================================================

        print("\n--- Local training ---")

        for client in self.clients:

            print(
                f"[Round {round_number + 1}] "
                f"Training client {client.id}"
            )

            client.train_local(
                epochs=self.local_epochs
            )

        # =========================================================
        # 2. Exchange messages
        # =========================================================

        print("\n--- Creating messages ---")

        messages = {}

        for client in self.clients:

            received = []

            # Own model
            received.append(
                client.create_message()
            )

            # Neighbor models
            for neighbor in self.network.neighbors(
                client.id
            ):

                received.append(
                    self.clients[neighbor].create_message()
                )

            messages[client.id] = received

        # =========================================================
        # 3. Compute ground-truth LCV
        # =========================================================

        print(
            "\n--- Computing Ground-Truth "
            "Local Contribution Vectors ---"
        )

        ground_truth_lcv_dict = {}

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

            ground_truth_lcv_dict[client.id] = vector

        self.history["lcv_vectors"].append(
            copy.deepcopy(ground_truth_lcv_dict)
        )

        # =========================================================
        # 4. Update every attack scenario
        # =========================================================

        print("\n--- Updating coordinators ---")

        for name, (malicious_ids, strength) in self.scenarios.items():

            # -----------------------------------------------------
            # Ground truth
            # -----------------------------------------------------
            #
            # Never modify this copy.
            #

            ground_truth = {
                cid: vec.clone()
                for cid, vec in ground_truth_lcv_dict.items()
            }

            # -----------------------------------------------------
            # Construct malicious reports
            # -----------------------------------------------------

            reported = {}

            for cid, vec in ground_truth_lcv_dict.items():

                v = vec.clone()

                if strength is not None and cid in malicious_ids:

                    # Current attack:
                    # malicious client inflates its own LCV.
                    v[cid] = strength

                reported[cid] = v

            # -----------------------------------------------------
            # Expected LCV for outlier detection
            # -----------------------------------------------------
            #
            # TEMPORARY EXPERIMENTAL VERSION:
            #
            # We use the ground-truth LCV as the expected value.
            #
            # This gives us a clean test of the complete defense:
            #
            # reported LCV
            #       ↓
            # outlier detection
            #       ↓
            # audit
            #       ↓
            # recompute/correct
            #       ↓
            # zero malicious reward
            #
            # In the final protocol this must be replaced by an
            # estimate available to the coordinator without
            # trusting the client's report.
            #

            expected = {
                cid: vec.clone()
                for cid, vec in ground_truth_lcv_dict.items()
            }

            # -----------------------------------------------------
            # Run coordinator
            # -----------------------------------------------------

            self.coordinators[name].update_round(
                ground_truth_lcv_dict=ground_truth,
                reported_lcv_dict=reported,
                network=self.network,
                expected_lcv_dict=expected,
            )

            # -----------------------------------------------------
            # Save contribution history
            # -----------------------------------------------------

            self.history["contributions"][name].append(
                copy.deepcopy(
                    self.coordinators[name].get_all_contributions()
                )
            )

            # -----------------------------------------------------
            # Save audit history
            # -----------------------------------------------------

            self.history["audit_logs"][name].append(
                copy.deepcopy(
                    self.coordinators[name].get_audit_log()[-1]
                )
            )

        print("Coordinator updates complete")

        # =========================================================
        # 5. Aggregate models
        # =========================================================
        #
        # IMPORTANT:
        #
        # The current experiments attack the LCV reports only.
        # Therefore model aggregation remains unchanged.
        #
        # The audited contribution vectors are being evaluated as
        # the defense signal, not yet used to modify aggregation.
        #

        print("\n--- Model aggregation ---")

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

            mean_accuracy = (
                sum(accuracies)
                / len(accuracies)
            )

            self.history["client_accuracy"].append(
                accuracies
            )

            self.history["accuracy"].append(
                mean_accuracy
            )

            print(
                f"Round {r + 1}/{self.rounds} "
                f"| Mean accuracy: {mean_accuracy:.4f}"
            )

        print(
            "Training finished"
        )

        return self.clients

    def get_history(self):

        return self.history