import torch
import copy
import random


class Coordinator:
    """
    Coordinator for TRIP-Shapley contribution propagation.

    Tracks three versions:

        original:
            No defense. Reported LCVs are propagated unchanged.

        modified:
            Baseline from the previous implementation. The client's
            own contribution is always removed.

        audited:
            Proposed defense.

            1. Collect all reported LCVs.
            2. Estimate the normal contribution of every client from
               the reports of the other clients.
            3. Detect reports that are sufficiently inconsistent
               with these estimates.
            4. Force suspicious reports to be audited.
            5. Additionally audit random reports with probability p.
            6. If an audited report is dishonest:
                   - recompute/correct it using the ground-truth LCV;
                   - remove the attacker's reward for this round.
            7. If the report is honest, keep it unchanged.

    The ground-truth LCV is supplied directly by the simulator.
    This is a simulation shortcut. In the real protocol, the
    coordinator would recompute the LCV from the exchanged models
    during an audit.
    """

    def __init__(
        self,
        num_clients,
        audit_probability=0.0,
        outlier_threshold=0.001,
        seed=None,
    ):
        self.num_clients = num_clients

        # Probability of a random audit.
        self.audit_probability = audit_probability

        # Maximum tolerated deviation from the estimated normal
        # contribution before a report becomes an outlier.
        self.outlier_threshold = outlier_threshold

        self.rng = random.Random(seed)

        # ---------------------------------------------------------
        # Contribution states
        # ---------------------------------------------------------

        self.original_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }

        self.modified_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }

        self.audited_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }

        # ---------------------------------------------------------
        # Audit history
        # ---------------------------------------------------------

        self.audit_log = []

        # History of estimated "normal" contributions.
        self.outlier_history = []

    # =============================================================
    # Contribution propagation
    # =============================================================

    def update_single(
        self,
        client_id,
        local_contribution_vector,
        network,
        contribution_state,
    ):
        """
        Performs one coordinator propagation update.
        """

        neighbors = network.neighbors(client_id)
        participants = [client_id] + list(neighbors)

        weights = network.get_weights(client_id)

        propagated = torch.zeros(self.num_clients)
        total_weight = 0.0

        for p in participants:
            w = weights[p]

            propagated += w * contribution_state[p]
            total_weight += w

        if total_weight > 0:
            propagated /= total_weight

        return propagated + local_contribution_vector

    # =============================================================
    # TRIP-Shapley-style outlier detection
    # =============================================================

    def _estimate_normal_contributions(
        self,
        reported_lcv_dict,
    ):
        """
        Estimate the normal contribution of every client.

        For client j, we collect all reported LCV entries that
        correspond to j.

        Honest clients should report approximately the same
        contribution for j. Therefore, the median is used as a
        robust estimate of the underlying contribution.

        The median is deliberately used instead of the mean:
        a small number of malicious reports cannot move the
        estimate as easily.

        Returns:
            estimated_contributions:
                Tensor of shape [num_clients].

            values_by_target:
                Dictionary containing all observed reports for
                every target client.
        """

        values_by_target = {
            j: []
            for j in range(self.num_clients)
        }

        # ---------------------------------------------------------
        # Collect every reported contribution.
        #
        # Only non-zero entries are considered, matching the
        # sparse nature of LCV reports in TRIP-Shapley.
        # ---------------------------------------------------------

        for reporter_id, lcv in reported_lcv_dict.items():

            for target_id in range(self.num_clients):

                value = float(lcv[target_id])

                if value != 0.0:
                    values_by_target[target_id].append(value)

        # ---------------------------------------------------------
        # Robust estimate for every target.
        # ---------------------------------------------------------

        estimated = torch.zeros(self.num_clients)

        for target_id in range(self.num_clients):

            values = values_by_target[target_id]

            if len(values) == 0:
                continue

            tensor_values = torch.tensor(values)

            estimated[target_id] = torch.median(
                tensor_values
            )

        return estimated, values_by_target

    def _detect_outliers(
        self,
        reported_lcv_dict,
    ):
        """
        Detect suspicious LCV reports.

        For each reported LCV entry:

            deviation =
                |reported value - estimated normal value|

        If the deviation exceeds `outlier_threshold`, that
        particular entry is considered suspicious.

        A client becomes an outlier if at least one of its reported
        entries is suspicious.

        Returns:

            client_outliers:
                {
                    client_id: bool
                }

            entry_outliers:
                {
                    client_id: {
                        target_id: bool
                    }
                }

            estimated_contributions:
                Robust estimate for every target client.

            max_deviations:
                Maximum deviation observed for every reporting client.
        """

        (
            estimated_contributions,
            values_by_target,
        ) = self._estimate_normal_contributions(
            reported_lcv_dict
        )

        client_outliers = {}
        entry_outliers = {}
        max_deviations = {}

        for reporter_id, lcv in reported_lcv_dict.items():

            entry_outliers[reporter_id] = {}

            max_deviation = 0.0
            client_is_outlier = False

            for target_id in range(self.num_clients):

                value = float(lcv[target_id])
                expected = float(
                    estimated_contributions[target_id]
                )

                # Do not test absent/zero entries against the
                # estimated contribution.
                if value == 0.0:
                    entry_outliers[reporter_id][target_id] = False
                    continue

                deviation = abs(value - expected)

                is_outlier = (
                    deviation > self.outlier_threshold
                )

                entry_outliers[reporter_id][target_id] = (
                    is_outlier
                )

                max_deviation = max(
                    max_deviation,
                    deviation
                )

                if is_outlier:
                    client_is_outlier = True

            client_outliers[reporter_id] = client_is_outlier
            max_deviations[reporter_id] = max_deviation

        return (
            client_outliers,
            entry_outliers,
            estimated_contributions,
            max_deviations,
        )

    # =============================================================
    # Probabilistic audit
    # =============================================================

    def _audit_client(
        self,
        client_id,
        ground_truth_lcv,
        reported_lcv,
        forced_audit,
        outlier_deviation,
    ):
        """
        Decide whether a client's LCV should be audited.

        An audit happens if:

            forced_audit
            OR
            random() < audit_probability

        If audited, compare the complete reported vector against
        the ground-truth vector.

        A dishonest report is replaced by the ground-truth vector,
        with the attacker's own reward set to zero.
        """

        random_audit = (
            self.rng.random()
            < self.audit_probability
        )

        audited = forced_audit or random_audit

        if forced_audit:
            audit_reason = "outlier"

        elif random_audit:
            audit_reason = "random"

        else:
            audit_reason = None

        # ---------------------------------------------------------
        # No audit
        # ---------------------------------------------------------

        if not audited:

            return (
                reported_lcv.clone(),
                {
                    "audited": False,
                    "accepted": None,
                    "flagged": False,
                    "audit_reason": None,
                    "max_deviation": None,
                    "outlier_deviation": outlier_deviation,
                },
            )

        # ---------------------------------------------------------
        # Audit: compare against deterministic ground truth.
        # ---------------------------------------------------------

        deviation = (
            reported_lcv - ground_truth_lcv
        ).abs()

        max_deviation = float(
            deviation.max()
        )

        # An audit checks whether the reported vector is exactly
        # the deterministic LCV.
        #
        # We use a tiny numerical tolerance because this is a
        # floating-point computation.
        numerical_tolerance = 1e-7

        accepted = (
            max_deviation <= numerical_tolerance
        )

        # ---------------------------------------------------------
        # Honest report
        # ---------------------------------------------------------

        if accepted:

            return (
                reported_lcv.clone(),
                {
                    "audited": True,
                    "accepted": True,
                    "flagged": False,
                    "audit_reason": audit_reason,
                    "max_deviation": max_deviation,
                    "outlier_deviation": outlier_deviation,
                },
            )

        # ---------------------------------------------------------
        # Dishonest report
        #
        # Replace it with the recomputed correct LCV.
        #
        # IMPORTANT:
        # We do NOT zero the LCV itself globally.
        #
        # We only remove the malicious client's reward for this
        # round.
        # ---------------------------------------------------------

        corrected_lcv = ground_truth_lcv.clone()

        corrected_lcv[client_id] = 0.0

        return (
            corrected_lcv,
            {
                "audited": True,
                "accepted": False,
                "flagged": True,
                "audit_reason": audit_reason,
                "max_deviation": max_deviation,
                "outlier_deviation": outlier_deviation,
            },
        )

    # =============================================================
    # Complete round
    # =============================================================

    def update_round(
        self,
        ground_truth_lcv_dict,
        reported_lcv_dict,
        network,
    ):
        """
        Update all contribution versions.

        Parameters
        ----------
        ground_truth_lcv_dict:
            Correct LCV computed before any malicious manipulation.

        reported_lcv_dict:
            LCVs actually reported by clients.

        network:
            DFL network.

        The audited version first performs global outlier detection
        across all reported LCVs. Suspicious reports are then
        automatically audited, while additional reports are
        randomly audited.
        """

        print(
            f"[Coordinator] Updating contributions for "
            f"{len(reported_lcv_dict)} clients"
        )

        # =========================================================
        # 1. Detect LCV outliers BEFORE propagating anything.
        # =========================================================

        (
            client_outliers,
            entry_outliers,
            estimated_contributions,
            max_deviations,
        ) = self._detect_outliers(
            reported_lcv_dict
        )

        # Save estimated contribution vector for analysis.
        self.outlier_history.append(
            {
                "estimated_contributions":
                    estimated_contributions.clone(),

                "client_outliers":
                    copy.deepcopy(client_outliers),

                "entry_outliers":
                    copy.deepcopy(entry_outliers),

                "max_deviations":
                    copy.deepcopy(max_deviations),
            }
        )

        # =========================================================
        # 2. Propagate all three versions.
        # =========================================================

        new_original = {}
        new_modified = {}
        new_audited = {}

        round_audit_log = {}

        for client_id, reported_lcv in reported_lcv_dict.items():

            ground_truth_lcv = (
                ground_truth_lcv_dict[client_id]
            )

            print(
                f"[Coordinator] Updating client {client_id}"
            )

            # -----------------------------------------------------
            # Version 1: original
            # -----------------------------------------------------

            new_original[client_id] = (
                self.update_single(
                    client_id,
                    reported_lcv,
                    network,
                    self.original_contributions,
                )
            )

            # -----------------------------------------------------
            # Version 2: previous modified baseline
            # -----------------------------------------------------

            modified_lcv = copy.deepcopy(
                reported_lcv
            )

            modified_lcv[client_id] = 0.0

            new_modified[client_id] = (
                self.update_single(
                    client_id,
                    modified_lcv,
                    network,
                    self.modified_contributions,
                )
            )

            # -----------------------------------------------------
            # Version 3: proposed audited protocol
            # -----------------------------------------------------

            forced_audit = client_outliers[
                client_id
            ]

            outlier_deviation = max_deviations[
                client_id
            ]

            audited_lcv, outcome = (
                self._audit_client(
                    client_id=client_id,
                    ground_truth_lcv=ground_truth_lcv,
                    reported_lcv=reported_lcv,
                    forced_audit=forced_audit,
                    outlier_deviation=outlier_deviation,
                )
            )

            round_audit_log[client_id] = outcome

            new_audited[client_id] = (
                self.update_single(
                    client_id,
                    audited_lcv,
                    network,
                    self.audited_contributions,
                )
            )

        # =========================================================
        # 3. Commit state.
        # =========================================================

        self.original_contributions = new_original
        self.modified_contributions = new_modified
        self.audited_contributions = new_audited

        self.audit_log.append(
            round_audit_log
        )

        print(
            "[Coordinator] Round propagation finished"
        )

    # =============================================================
    # Accessors
    # =============================================================

    def get_contribution(
        self,
        client_id,
        method="original",
    ):
        mapping = {
            "original":
                self.original_contributions,

            "modified":
                self.modified_contributions,

            "audited":
                self.audited_contributions,
        }

        if method not in mapping:
            raise ValueError(
                f"Unknown method: {method}"
            )

        return mapping[method][client_id]

    def get_all_contributions(self):
        return {
            "original":
                self.original_contributions,

            "modified":
                self.modified_contributions,

            "audited":
                self.audited_contributions,
        }

    def get_audit_log(self):
        return self.audit_log

    def get_outlier_history(self):
        return self.outlier_history