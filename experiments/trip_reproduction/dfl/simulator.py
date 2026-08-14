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

    The expensive part of the experiment is:

        local training
            +
        LCV computation

    These are performed ONCE per seed.

    After the ground-truth LCVs have been computed, many
    independent coordinator configurations can be evaluated:

        audit probability p
        outlier threshold
        attack scenario

    This means that sweeping over probabilities and thresholds
    does NOT require repeating model training or LCV computation.

    Each coordinator maintains:

        original:
            No defense. Propagates the reported LCV unchanged.

        modified:
            Baseline defense. Removes the client's own
            contribution before propagation.

        audited:
            Probabilistic + optional outlier-triggered
            verification.

    IMPORTANT:

    The ground-truth LCV is available to the simulator because
    this is an experimental simulation.

    In the actual protocol, the coordinator would recompute
    the LCV from the exchanged models when an audit is triggered.
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

        # -----------------------------------------------------
        # Experiment sweeps
        # -----------------------------------------------------

        audit_probabilities=None,
        outlier_thresholds=None,

        seed=42,
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device

        self.single_attacker_id = single_attacker_id
        self.multi_attacker_ids = multi_attacker_ids or []

        self.seed = seed

        # -----------------------------------------------------
        # Sweep parameters
        # -----------------------------------------------------

        if audit_probabilities is None:
            audit_probabilities = [0.1]

        if outlier_thresholds is None:
            outlier_thresholds = [0.001]

        self.audit_probabilities = list(
            audit_probabilities
        )

        self.outlier_thresholds = list(
            outlier_thresholds
        )

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------

        client_loaders, test_loader = create_client_loaders(
            num_clients=num_clients,
            batch_size=batch_size
        )

        self.test_loader = test_loader

        # -----------------------------------------------------
        # LCV function
        # -----------------------------------------------------

        lcv_function = compute_lcv

        # -----------------------------------------------------
        # Create synchronized clients
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Network
        # -----------------------------------------------------

        self.network = Network(
            num_clients=num_clients,
            topology=topology
        )

        # -----------------------------------------------------
        # Attack scenarios
        # -----------------------------------------------------
        #
        # The stealth attacks are special:
        #
        #   stealth_half:
        #       reported own contribution is increased by
        #       threshold / 2
        #
        #   stealth_full:
        #       reported own contribution is increased by
        #       exactly threshold
        #
        # Their actual attack strength is therefore determined
        # dynamically for each threshold configuration (this is
        # intentional: the attack scales with whatever threshold
        # the defense is currently using, rather than a fixed
        # reference value).
        #
        # -----------------------------------------------------

        self.scenarios = {

            "clean": {
                "type": "clean",
                "attacker_ids": set(),
                "strength": None,
            },

            "single_s1": {
                "type": "fixed",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": 1.0,
            },

            "single_s5": {
                "type": "fixed",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": 5.0,
            },

            "single_s10": {
                "type": "fixed",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": 10.0,
            },

            "single_s20": {
                "type": "fixed",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": 20.0,
            },

            "single_s50": {
                "type": "fixed",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": 50.0,
            },

            "multi_2": {
                "type": "fixed",
                "attacker_ids": set(
                    self.multi_attacker_ids[:2]
                ),
                "strength": 1.0,
            },

            "multi_3": {
                "type": "fixed",
                "attacker_ids": set(
                    self.multi_attacker_ids[:3]
                ),
                "strength": 1.0,
            },

            "stealth_half": {
                "type": "stealth_half",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": None,
            },

            "stealth_full": {
                "type": "stealth_full",
                "attacker_ids": {
                    self.single_attacker_id
                },
                "strength": None,
            },
        }

        # -----------------------------------------------------
        # Coordinators
        # -----------------------------------------------------
        #
        # One coordinator for EVERY:
        #
        #   scenario × audit_probability × threshold
        #
        # combination.
        #
        # These coordinators are independent, but all receive
        # exactly the same ground-truth LCVs.
        #
        # Therefore we do NOT repeat training or LCV computation.
        #
        # -----------------------------------------------------

        self.coordinators = {}

        coordinator_index = 0

        for p in self.audit_probabilities:

            for threshold in self.outlier_thresholds:

                for scenario_name in self.scenarios:

                    key = self._make_config_key(
                        scenario_name,
                        p,
                        threshold,
                    )

                    # Give every configuration an independent
                    # deterministic random stream.
                    coordinator_seed = (
                        seed
                        + coordinator_index
                    )

                    coordinator_index += 1

                    self.coordinators[key] = Coordinator(
                        num_clients=num_clients,
                        audit_probability=p,
                        outlier_threshold=threshold,
                        seed=coordinator_seed,
                    )

        # -----------------------------------------------------
        # History
        # -----------------------------------------------------

        self.history = {

            # Accuracy only needs to be stored once because
            # every coordinator configuration uses the exact
            # same model training.
            "accuracy": [],
            "client_accuracy": [],

            # Ground-truth LCVs.
            "lcv_vectors": [],

            # Contribution history for every configuration.
            "contributions": {
                key: []
                for key in self.coordinators
            },

            # Audit history for every configuration.
            "audit_logs": {
                key: []
                for key in self.coordinators
            },

            # Outlier detection history.
            "outlier_history": {
                key: []
                for key in self.coordinators
            },

            # Experiment metadata.
            "topology": topology,
            "num_clients": num_clients,
            "rounds": rounds,

            "single_attacker_id":
                self.single_attacker_id,

            "multi_attacker_ids":
                self.multi_attacker_ids,

            "audit_probabilities":
                self.audit_probabilities,

            "outlier_thresholds":
                self.outlier_thresholds,

            "seed": seed,
        }

    # =========================================================
    # Configuration helpers
    # =========================================================

    @staticmethod
    def _make_config_key(
        scenario_name,
        audit_probability,
        outlier_threshold,
    ):
        """
        Create a unique identifier for one experimental
        configuration.
        """

        threshold_string = (
            "None"
            if outlier_threshold is None
            else str(outlier_threshold)
        )

        return (
            f"{scenario_name}"
            f"__p_{audit_probability}"
            f"__threshold_{threshold_string}"
        )

    # =========================================================
    # Determine attack strength
    # =========================================================

    def _get_attack_strength(
        self,
        scenario,
        outlier_threshold,
        honest_value,
    ):
        """
        Determine the malicious value for one attacker.

        Fixed attacks:
            use the predefined absolute value.

        stealth_half:
            honest value + threshold / 2

        stealth_full:
            honest value + threshold

        If threshold is None, stealth attacks are disabled
        because there is no threshold-relative attack magnitude.

        NOTE: by design, the stealth attack magnitude tracks
        whichever outlier_threshold is currently being swept,
        rather than a fixed reference threshold.
        """

        scenario_type = scenario["type"]

        # -----------------------------------------------------
        # Clean
        # -----------------------------------------------------

        if scenario_type == "clean":
            return honest_value

        # -----------------------------------------------------
        # Fixed-strength attack
        # -----------------------------------------------------

        if scenario_type == "fixed":
            return scenario["strength"]

        # -----------------------------------------------------
        # Stealth attacks
        # -----------------------------------------------------

        if outlier_threshold is None:
            return honest_value

        if scenario_type == "stealth_half":
            return (
                honest_value
                + outlier_threshold / 2.0
            )

        if scenario_type == "stealth_full":
            return (
                honest_value
                + outlier_threshold
            )

        raise ValueError(
            f"Unknown scenario type: {scenario_type}"
        )

    # =========================================================
    # Construct reports
    # =========================================================

    def _construct_reports(
        self,
        ground_truth_lcv_dict,
        scenario,
        outlier_threshold,
    ):
        """
        Construct the reported LCV vectors for one attack
        scenario and one threshold.

        Ground truth is NEVER modified.
        """

        reported = {}

        malicious_ids = scenario["attacker_ids"]

        for cid, vec in ground_truth_lcv_dict.items():

            v = vec.clone()

            if cid in malicious_ids:

                honest_value = float(
                    vec[cid]
                )

                attack_value = (
                    self._get_attack_strength(
                        scenario=scenario,
                        outlier_threshold=outlier_threshold,
                        honest_value=honest_value,
                    )
                )

                v[cid] = attack_value

            reported[cid] = v

        return reported

    # =========================================================
    # One training round
    # =========================================================

    def train_round(self, round_number):

        print("\n======================")
        print(
            f"Starting round "
            f"{round_number + 1}"
        )
        print("======================")

        # =====================================================
        # 1. Local training
        # =====================================================

        print("\n--- Local training ---")

        for client in self.clients:

            print(
                f"[Round {round_number + 1}] "
                f"Training client {client.id}"
            )

            client.train_local(
                epochs=self.local_epochs
            )

        # =====================================================
        # 2. Exchange messages
        # =====================================================

        print("\n--- Creating messages ---")

        messages = {}

        for client in self.clients:

            received = []

            # Own model.
            received.append(
                client.create_message()
            )

            # Neighbor models.
            for neighbor in self.network.neighbors(
                client.id
            ):

                received.append(
                    self.clients[neighbor].create_message()
                )

            messages[client.id] = received

        # =====================================================
        # 3. Compute ground-truth LCV
        # =====================================================
        #
        # THIS IS THE EXPENSIVE PART.
        #
        # It is performed exactly once per client.
        #
        # Every p / threshold / scenario configuration below
        # reuses this same result.
        #
        # =====================================================

        print(
            "\n--- Computing Ground-Truth "
            "Local Contribution Vectors ---"
        )

        ground_truth_lcv_dict = {}

        for client in self.clients:

            print(
                f"\n[Client {client.id}] "
                f"Starting LCV"
            )

            lcv = client.compute_lcv(
                received_messages=messages[
                    client.id
                ],
                test_loader=self.test_loader
            )

            print(
                f"[Client {client.id}] "
                f"Finished LCV"
            )

            vector = torch.zeros(
                self.num_clients
            )

            for cid, value in lcv.items():
                vector[cid] = value

            ground_truth_lcv_dict[
                client.id
            ] = vector

        # Store ground truth once.
        self.history[
            "lcv_vectors"
        ].append(
            copy.deepcopy(
                ground_truth_lcv_dict
            )
        )

        # =====================================================
        # 4. Evaluate ALL coordinator configurations
        # =====================================================
        #
        # No model training happens here.
        #
        # No LCV computation happens here.
        #
        # We only perform:
        #
        #   - report construction
        #   - outlier detection
        #   - random audit decisions
        #   - contribution propagation
        #
        # IMPORTANT: reported LCVs depend only on
        # (scenario, threshold) -- NOT on audit probability p,
        # since the stealth attack magnitude is a function of
        # the threshold, not of p. So reports are constructed
        # once per (scenario, threshold) and then reused across
        # every p in the sweep, instead of being rebuilt inside
        # the p loop.
        #
        # =====================================================

        print(
            "\n--- Evaluating coordinator "
            "configurations ---"
        )

        for threshold in self.outlier_thresholds:

            # -------------------------------------------------
            # Build reports for every scenario ONCE for this
            # threshold. These are reused for every p below.
            # -------------------------------------------------

            reported_by_scenario = {}

            for scenario_name, scenario in (
                self.scenarios.items()
            ):

                reported_by_scenario[
                    scenario_name
                ] = self._construct_reports(
                    ground_truth_lcv_dict=
                        ground_truth_lcv_dict,

                    scenario=scenario,

                    outlier_threshold=
                        threshold,
                )

            for p in self.audit_probabilities:

                print(
                    f"\n[p={p}, "
                    f"threshold={threshold}]"
                )

                for scenario_name, scenario in (
                    self.scenarios.items()
                ):

                    key = self._make_config_key(
                        scenario_name,
                        p,
                        threshold,
                    )

                    print(
                        f"  Scenario: "
                        f"{scenario_name}"
                    )

                    # -----------------------------------------
                    # Ground truth
                    # -----------------------------------------

                    ground_truth = {
                        cid: vec.clone()
                        for cid, vec
                        in ground_truth_lcv_dict.items()
                    }

                    # -----------------------------------------
                    # Reuse the reports built above.
                    # -----------------------------------------

                    reported = (
                        reported_by_scenario[
                            scenario_name
                        ]
                    )

                    # -----------------------------------------
                    # Update coordinator
                    # -----------------------------------------

                    coordinator = (
                        self.coordinators[key]
                    )

                    coordinator.update_round(
                        ground_truth_lcv_dict=
                            ground_truth,

                        reported_lcv_dict=
                            reported,

                        network=self.network,
                    )

                    # -----------------------------------------
                    # Save contributions
                    # -----------------------------------------

                    self.history[
                        "contributions"
                    ][key].append(
                        copy.deepcopy(
                            coordinator.get_all_contributions()
                        )
                    )

                    # -----------------------------------------
                    # Save audit history
                    # -----------------------------------------

                    audit_log = (
                        coordinator.get_audit_log()
                    )

                    self.history[
                        "audit_logs"
                    ][key].append(
                        copy.deepcopy(
                            audit_log[-1]
                        )
                    )

                    # -----------------------------------------
                    # Save outlier history
                    # -----------------------------------------

                    outlier_history = (
                        coordinator.get_outlier_history()
                    )

                    self.history[
                        "outlier_history"
                    ][key].append(
                        copy.deepcopy(
                            outlier_history[-1]
                        )
                    )

        print(
            "\nAll coordinator configurations "
            "updated."
        )

        # =====================================================
        # 5. Model aggregation
        # =====================================================
        #
        # IMPORTANT:
        #
        # The attacks currently manipulate LCV reports only.
        #
        # Therefore model aggregation remains unchanged.
        #
        # Every configuration sees the same model trajectory.
        #
        # =====================================================

        print("\n--- Model aggregation ---")

        for client in self.clients:

            weights = self.network.get_weights(
                client.id
            )

            client.aggregate(
                received_messages=messages[
                    client.id
                ],
                weights=weights
            )

        print(
            f"Round {round_number + 1} complete"
        )

    # =========================================================
    # Evaluation
    # =========================================================

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

    # =========================================================
    # Train
    # =========================================================

    def train(self):

        print(
            "Starting DFL training"
        )

        print(
            f"Audit probabilities: "
            f"{self.audit_probabilities}"
        )

        print(
            f"Outlier thresholds: "
            f"{self.outlier_thresholds}"
        )

        print(
            f"Coordinator configurations: "
            f"{len(self.coordinators)}"
        )

        self.network.print_network()

        # -----------------------------------------------------
        # Training loop
        # -----------------------------------------------------

        for r in range(self.rounds):

            self.train_round(r)

            # -------------------------------------------------
            # Evaluation
            # -------------------------------------------------

            print(
                "\n--- Evaluation ---"
            )

            accuracies = self.evaluate()

            mean_accuracy = (
                sum(accuracies)
                / len(accuracies)
            )

            self.history[
                "client_accuracy"
            ].append(
                accuracies
            )

            self.history[
                "accuracy"
            ].append(
                mean_accuracy
            )

            print(
                f"Round {r + 1}/{self.rounds} "
                f"| Mean accuracy: "
                f"{mean_accuracy:.4f}"
            )

        print(
            "\nTraining finished"
        )

        return self.clients

    # =========================================================
    # History
    # =========================================================

    def get_history(self):

        return self.history