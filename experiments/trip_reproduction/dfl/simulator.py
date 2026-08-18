import inspect
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

    This means that sweeping over probabilities and thresholds does
    NOT require repeating model training or LCV computation.

    Each coordinator maintains:

        original:
            No defense. Propagates the reported LCV unchanged.

        audited:
            Probabilistic + optional outlier-triggered
            verification.

    IMPORTANT:

    The ground-truth LCV is available to the simulator because this
    is an experimental simulation. In the actual protocol, the
    coordinator would recompute the LCV from the exchanged models
    when an audit is triggered.
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
        audit_threshold=1e-6,

        seed=42,
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.topology = topology
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

        self.audit_probabilities = list(audit_probabilities)
        self.outlier_thresholds = list(outlier_thresholds)

        # Numerical-precision tolerance used by every coordinator's
        # accept/reject test during an audit. NOT the economically
        # derived threshold - see Coordinator's docstring.
        self.audit_threshold = audit_threshold

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------
        #
        # Pass `seed` through to create_client_loaders if (and
        # only if) that function actually accepts a seed parameter,
        # so this doesn't crash regardless of dataset.py's exact
        # signature. Checked defensively via introspection since
        # the signature isn't hardcoded here.
        # -----------------------------------------------------

        loader_kwargs = dict(
            num_clients=num_clients,
            batch_size=batch_size,
        )

        try:
            loader_params = inspect.signature(
                create_client_loaders
            ).parameters
        except (TypeError, ValueError):
            loader_params = {}

        if "seed" in loader_params:
            loader_kwargs["seed"] = seed
            print(
                "[DFLSimulator] Passing seed="
                f"{seed} to create_client_loaders."
            )
        else:
            print(
                "[DFLSimulator] create_client_loaders does not "
                "accept a 'seed' parameter - dataset partitioning "
                "relies on the global RNG state set before "
                "construction. Verify this is deterministic if "
                "reproducibility matters."
            )

        client_loaders, test_loader = create_client_loaders(
            **loader_kwargs
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

            client.model.load_state_dict(global_state)

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
        # stealth_half:
        #     reported own contribution is increased by
        #     threshold / 2
        #
        # stealth_full:
        #     reported own contribution is increased by
        #     exactly threshold
        #
        # Their attack strength is therefore determined
        # dynamically for each threshold configuration. When
        # threshold is None, there is no reference value to scale
        # from - these two scenario/threshold combinations are
        # SKIPPED entirely (no coordinator created, no history
        # entries), rather than silently falling back to "no
        # attack" (which would make them indistinguishable from
        # "clean" in the saved results). See
        # _is_scenario_threshold_active.
        # -----------------------------------------------------

        self.scenarios = {

            "clean": {
                "type": "clean",
                "attacker_ids": set(),
                "strength": None,
            },

            "single_s1": {
                "type": "fixed",
                "attacker_ids": {self.single_attacker_id},
                "strength": 1.0,
            },

            "single_s5": {
                "type": "fixed",
                "attacker_ids": {self.single_attacker_id},
                "strength": 5.0,
            },

            "single_s10": {
                "type": "fixed",
                "attacker_ids": {self.single_attacker_id},
                "strength": 10.0,
            },

            "single_s20": {
                "type": "fixed",
                "attacker_ids": {self.single_attacker_id},
                "strength": 20.0,
            },

            "single_s50": {
                "type": "fixed",
                "attacker_ids": {self.single_attacker_id},
                "strength": 50.0,
            },

            "multi_2": {
                "type": "fixed",
                "attacker_ids": set(self.multi_attacker_ids[:2]),
                "strength": 1.0,
            },

            "multi_3": {
                "type": "fixed",
                "attacker_ids": set(self.multi_attacker_ids[:3]),
                "strength": 1.0,
            },

            "stealth_half": {
                "type": "stealth_half",
                "attacker_ids": {self.single_attacker_id},
                "strength": None,
            },

            "stealth_full": {
                "type": "stealth_full",
                "attacker_ids": {self.single_attacker_id},
                "strength": None,
            },
        }

        # -----------------------------------------------------
        # Coordinators
        # -----------------------------------------------------
        #
        # One coordinator for every ACTIVE:
        #
        #   scenario x audit_probability x threshold
        #
        # combination. stealth_half/stealth_full x threshold=None
        # combinations are skipped (see docstring above) and
        # recorded in self.skipped_configs for transparency.
        #
        # All coordinators receive exactly the same ground-truth
        # LCVs each round, so training/LCV computation is never
        # repeated across configurations.
        # -----------------------------------------------------

        self.coordinators = {}
        self.skipped_configs = []

        coordinator_index = 0

        for p in self.audit_probabilities:

            for threshold in self.outlier_thresholds:

                for scenario_name, scenario in (
                    self.scenarios.items()
                ):

                    key = self._make_config_key(
                        scenario_name, p, threshold
                    )

                    if not self._is_scenario_threshold_active(
                        scenario, threshold
                    ):
                        self.skipped_configs.append(key)
                        continue

                    coordinator_seed = seed + coordinator_index
                    coordinator_index += 1

                    self.coordinators[key] = Coordinator(
                        num_clients=num_clients,
                        audit_probability=p,
                        outlier_threshold=threshold,
                        audit_threshold=self.audit_threshold,
                        seed=coordinator_seed,
                    )

        if self.skipped_configs:
            print(
                f"[DFLSimulator] Skipped "
                f"{len(self.skipped_configs)} stealth "
                f"configurations with threshold=None (no "
                f"reference magnitude to attack relative to): "
                f"{self.skipped_configs}"
            )

        # -----------------------------------------------------
        # History
        # -----------------------------------------------------

        # Scenario metadata in a pickle-friendly form (sets ->
        # sorted lists) for full reproducibility of what was run.
        scenarios_serializable = {
            name: {
                "type": s["type"],
                "attacker_ids": sorted(s["attacker_ids"]),
                "strength": s["strength"],
            }
            for name, s in self.scenarios.items()
        }

        self.history = {

            "accuracy": [],
            "client_accuracy": [],

            "lcv_vectors": [],

            # Mean of the ground-truth self-contribution diagonal
            # (psi_i^(t)(i) averaged over clients), one entry per
            # round. Used to empirically estimate the honest
            # reward baseline "e" post-hoc (see
            # compute_theoretical_thresholds), and useful on its
            # own as a training-dynamics diagnostic.
            "ground_truth_diagonal_mean": [],

            "contributions": {
                key: [] for key in self.coordinators
            },

            "audit_logs": {
                key: [] for key in self.coordinators
            },

            "outlier_history": {
                key: [] for key in self.coordinators
            },

            "skipped_configs": self.skipped_configs,

            # ---------------------------------------------------
            # Full experimental configuration, saved for
            # reproducibility. This run cannot be repeated or
            # adjusted afterward, so everything needed to
            # reconstruct exactly what was run is captured here
            # rather than relying on external notes.
            # ---------------------------------------------------

            "topology": topology,
            "num_clients": num_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "device": str(device),

            "single_attacker_id": self.single_attacker_id,
            "multi_attacker_ids": self.multi_attacker_ids,

            "audit_probabilities": self.audit_probabilities,
            "outlier_thresholds": self.outlier_thresholds,
            "audit_threshold": self.audit_threshold,

            "scenarios": scenarios_serializable,

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

    @staticmethod
    def _is_scenario_threshold_active(scenario, outlier_threshold):
        """
        Returns False for stealth_half/stealth_full combined with
        threshold=None, since there is no reference magnitude for
        the attack in that case (see class docstring). All other
        combinations are always active.
        """

        if (
            scenario["type"] in ("stealth_half", "stealth_full")
            and outlier_threshold is None
        ):
            return False

        return True

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

        Fixed attacks: use the predefined absolute value.
        stealth_half: honest value + threshold / 2
        stealth_full: honest value + threshold

        Callers must check _is_scenario_threshold_active before
        reaching this point for stealth scenarios - the
        outlier_threshold is None fallback below is a defensive
        safety net and should be unreachable in normal operation.
        """

        scenario_type = scenario["type"]

        if scenario_type == "clean":
            return honest_value

        if scenario_type == "fixed":
            return scenario["strength"]

        if outlier_threshold is None:
            # Should be unreachable - _is_scenario_threshold_active
            # skips this combination upstream. Defensive fallback
            # only.
            return honest_value

        if scenario_type == "stealth_half":
            return honest_value + outlier_threshold / 2.0

        if scenario_type == "stealth_full":
            return honest_value + outlier_threshold

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
        Construct the reported LCV vectors for one attack scenario
        and one threshold. Ground truth is never modified.
        """

        reported = {}

        malicious_ids = scenario["attacker_ids"]

        for cid, vec in ground_truth_lcv_dict.items():

            v = vec.clone()

            if cid in malicious_ids:

                honest_value = float(vec[cid])

                attack_value = self._get_attack_strength(
                    scenario=scenario,
                    outlier_threshold=outlier_threshold,
                    honest_value=honest_value,
                )

                v[cid] = attack_value

            reported[cid] = v

        # Sanity check: the clean scenario must never differ from
        # ground truth. Cheap, only runs for one scenario, catches
        # a real bug immediately rather than producing silently
        # wrong false-positive-rate results.
        if scenario["type"] == "clean":
            for cid in reported:
                if not torch.equal(
                    reported[cid], ground_truth_lcv_dict[cid]
                ):
                    raise RuntimeError(
                        "Clean scenario report does not match "
                        f"ground truth for client {cid} - this "
                        "indicates a bug in report construction."
                    )

        return reported

    # =========================================================
    # One training round
    # =========================================================

    def train_round(self, round_number):

        print("\n======================")
        print(f"Starting round {round_number + 1}")
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

            client.train_local(epochs=self.local_epochs)

        # =====================================================
        # 2. Exchange messages
        # =====================================================

        print("\n--- Creating messages ---")

        messages = {}

        for client in self.clients:

            received = [client.create_message()]

            for neighbor in self.network.neighbors(client.id):

                received.append(
                    self.clients[neighbor].create_message()
                )

            messages[client.id] = received

        # =====================================================
        # 3. Compute ground-truth LCV
        # =====================================================
        #
        # THIS IS THE EXPENSIVE PART. Performed exactly once per
        # client. Every p / threshold / scenario configuration
        # below reuses this same result.
        # =====================================================

        print(
            "\n--- Computing Ground-Truth "
            "Local Contribution Vectors ---"
        )

        ground_truth_lcv_dict = {}

        for client in self.clients:

            lcv = client.compute_lcv(
                received_messages=messages[client.id],
                test_loader=self.test_loader
            )

            print(f"[Client {client.id}] Finished LCV")

            vector = torch.zeros(self.num_clients)

            for cid, value in lcv.items():
                vector[cid] = value

            ground_truth_lcv_dict[client.id] = vector

        self.history["lcv_vectors"].append(
            copy.deepcopy(ground_truth_lcv_dict)
        )

        diagonal_values = [
            float(ground_truth_lcv_dict[cid][cid])
            for cid in ground_truth_lcv_dict
        ]

        self.history["ground_truth_diagonal_mean"].append(
            sum(diagonal_values) / len(diagonal_values)
        )

        # =====================================================
        # 4. Evaluate ALL active coordinator configurations
        # =====================================================
        #
        # No model training happens here. No LCV computation
        # happens here. Only: report construction, outlier
        # detection, random audit decisions, contribution
        # propagation.
        #
        # Reported LCVs depend only on (scenario, threshold) - NOT
        # on audit probability p, since stealth attack magnitude
        # is a function of the threshold, not of p. Reports are
        # built once per (scenario, threshold) and reused across
        # every p in the sweep.
        # =====================================================

        print("\n--- Evaluating coordinator configurations ---")

        for threshold in self.outlier_thresholds:

            reported_by_scenario = {}

            for scenario_name, scenario in self.scenarios.items():

                if not self._is_scenario_threshold_active(
                    scenario, threshold
                ):
                    continue

                reported_by_scenario[scenario_name] = (
                    self._construct_reports(
                        ground_truth_lcv_dict=ground_truth_lcv_dict,
                        scenario=scenario,
                        outlier_threshold=threshold,
                    )
                )

            for p in self.audit_probabilities:

                for scenario_name, scenario in (
                    self.scenarios.items()
                ):

                    if not self._is_scenario_threshold_active(
                        scenario, threshold
                    ):
                        continue

                    key = self._make_config_key(
                        scenario_name, p, threshold
                    )

                    ground_truth = {
                        cid: vec.clone()
                        for cid, vec in
                        ground_truth_lcv_dict.items()
                    }

                    reported = reported_by_scenario[scenario_name]

                    coordinator = self.coordinators[key]

                    coordinator.update_round(
                        ground_truth_lcv_dict=ground_truth,
                        reported_lcv_dict=reported,
                        network=self.network,
                    )

                    self.history["contributions"][key].append(
                        copy.deepcopy(
                            coordinator.get_all_contributions()
                        )
                    )

                    audit_log = coordinator.get_audit_log()
                    self.history["audit_logs"][key].append(
                        copy.deepcopy(audit_log[-1])
                    )

                    outlier_history = (
                        coordinator.get_outlier_history()
                    )
                    self.history["outlier_history"][key].append(
                        copy.deepcopy(outlier_history[-1])
                    )

        print("\nAll coordinator configurations updated.")

        # =====================================================
        # 5. Model aggregation
        # =====================================================
        #
        # The attacks manipulate LCV reports only, so model
        # aggregation is unaffected and identical across every
        # configuration.
        # =====================================================

        print("\n--- Model aggregation ---")

        for client in self.clients:

            weights = self.network.get_weights(client.id)

            client.aggregate(
                received_messages=messages[client.id],
                weights=weights
            )

        print(f"Round {round_number + 1} complete")

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

        print("Starting DFL training")

        self.network.print_network()

        for r in range(self.rounds):

            self.train_round(r)

            print("\n--- Evaluation ---")

            accuracies = self.evaluate()

            mean_accuracy = sum(accuracies) / len(accuracies)

            self.history["client_accuracy"].append(accuracies)
            self.history["accuracy"].append(mean_accuracy)

            print(
                f"Round {r + 1}/{self.rounds} "
                f"| Mean accuracy: {mean_accuracy:.4f}"
            )

        print("\nTraining finished")

        return self.clients

    # =========================================================
    # Post-training analysis helpers
    # =========================================================

    def compute_theoretical_thresholds(self):
        """
        Computes, for every audit probability p, the two variants
        of the derived economic threshold:

            t <= p * e / (n * (1 - p))

        using:
            e = honest_reward_estimate, the mean of
                ground_truth_diagonal_mean across all rounds run
                so far.
            n = num_clients ("without neighbors" variant)
            n = average participant-set size, i.e.
                1 + average number of neighbors
                ("with neighbors" variant)

        These are THEORETICAL REFERENCE VALUES ONLY - never used as
        an actual accept/reject boundary anywhere in Coordinator
        (see its docstring). Results are stored into self.history
        under "honest_reward_estimate", "theoretical_thresholds",
        "n_neighbors_avg", "n_total".

        At p=1.0 the bound is vacuous (division by zero in the
        denominator, since any threshold satisfies it when the
        client is guaranteed to be audited) - stored as None for
        that probability rather than raising.

        Call this after train() so ground_truth_diagonal_mean has
        at least one entry.
        """

        diagonal_means = self.history["ground_truth_diagonal_mean"]

        if not diagonal_means:
            raise RuntimeError(
                "No ground-truth diagonal data recorded yet - "
                "call after train()."
            )

        e_estimate = sum(diagonal_means) / len(diagonal_means)

        n_total = self.num_clients

        participant_sizes = [
            1 + len(self.network.neighbors(i))
            for i in range(self.num_clients)
        ]
        n_neighbors_avg = (
            sum(participant_sizes) / len(participant_sizes)
        )

        thresholds = {}

        for p in self.audit_probabilities:

            if p >= 1.0:
                thresholds[p] = {
                    "with_neighbors": None,
                    "without_neighbors": None,
                }
                continue

            thresholds[p] = {
                "with_neighbors": (
                    (p * e_estimate)
                    / (n_neighbors_avg * (1 - p))
                ),
                "without_neighbors": (
                    (p * e_estimate)
                    / (n_total * (1 - p))
                ),
            }

        self.history["honest_reward_estimate"] = e_estimate
        self.history["theoretical_thresholds"] = thresholds
        self.history["n_neighbors_avg"] = n_neighbors_avg
        self.history["n_total"] = n_total

        return thresholds

    def compute_summary_metrics(self):
        """
        Computes and stores per-configuration summary metrics
        (audit counts, detection rates, false-positive rates, and
        final-round self-contribution values) directly into
        self.history["summary_metrics"], so downstream analysis
        does not need to re-derive them from the raw audit/outlier
        logs.

        Call this after train().
        """

        summary = {}

        for key, coordinator in self.coordinators.items():

            audit_log = coordinator.get_audit_log()
            outlier_history = coordinator.get_outlier_history()

            total_audits = 0
            total_random = 0
            total_outlier = 0
            total_both = 0
            total_flagged = 0

            for round_log in audit_log:
                for outcome in round_log.values():

                    if outcome["audited"]:
                        total_audits += 1

                    if outcome.get("random_audit", False):
                        total_random += 1

                    if outcome.get("outlier_audit", False):
                        total_outlier += 1

                    if outcome.get("audit_reason") == "both":
                        total_both += 1

                    if outcome["flagged"]:
                        total_flagged += 1

            total_outlier_flags = 0
            for round_info in outlier_history:
                total_outlier_flags += sum(
                    round_info["client_outliers"].values()
                )

            num_opportunities = (
                len(audit_log) * self.num_clients
                if audit_log else 0
            )

            audit_rate = (
                total_audits / num_opportunities
                if num_opportunities else 0.0
            )

            # NOTE: only meaningful as a true false-positive rate
            # for scenarios with zero actual attackers (e.g.
            # "clean") - for attacked scenarios this counts outlier
            # flags on the honest majority AND the attacker(s)
            # together. Use report_clean_false_positive_rates for
            # the scenario-specific, unambiguous version.
            outlier_flag_rate = (
                total_outlier_flags / num_opportunities
                if num_opportunities else 0.0
            )

            contributions = self.history["contributions"].get(
                key, []
            )

            final_original = None
            final_audited = None

            if contributions:
                final = contributions[-1]
                final_original = {
                    cid: float(vec[cid])
                    for cid, vec in final["original"].items()
                }
                final_audited = {
                    cid: float(vec[cid])
                    for cid, vec in final["audited"].items()
                }

            summary[key] = {
                "total_audits": total_audits,
                "total_random": total_random,
                "total_outlier": total_outlier,
                "total_both": total_both,
                "total_flagged": total_flagged,
                "total_outlier_flags": total_outlier_flags,
                "audit_rate": audit_rate,
                "outlier_flag_rate": outlier_flag_rate,
                "final_self_contribution_original": final_original,
                "final_self_contribution_audited": final_audited,
            }

        self.history["summary_metrics"] = summary

        return summary

    # =========================================================
    # History
    # =========================================================

    def get_history(self):

        return self.history