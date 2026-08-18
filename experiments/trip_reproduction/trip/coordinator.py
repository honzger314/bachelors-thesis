import torch
import copy
import random


class Coordinator:
    """
    Coordinator for TRIP-Shapley contribution propagation.

    Tracks two versions:

        original:
            No defense.
            Propagates the reported LCV unchanged.

        audited:
            Proposed defense.

            1. Collect reported LCVs.
            2. Optionally detect suspicious reports using the
               outlier heuristic.
            3. Force suspicious reports to be audited.
            4. Additionally audit random reports with probability p.
            5. If an audited report is dishonest:
                   - replace it with the ground-truth LCV;
                   - remove the attacker's reward for this round.
            6. If the report is honest, keep it unchanged.

    Parameters
    ----------
    num_clients:
        Number of clients in the DFL network.

    audit_probability:
        Probability p of a random audit. p=0 disables random
        auditing entirely (only the outlier heuristic, if enabled,
        can trigger an audit).

    outlier_threshold:
        Threshold used by the heuristic outlier detector.

        None:
            Disable outlier detection entirely.

        Otherwise:
            A report is considered suspicious if at least one
            reported contribution differs from the robust estimated
            normal contribution by more than this threshold.

    audit_threshold:
        Acceptance tolerance used when comparing a reported LCV
        against the ground truth during an audit.

        IMPORTANT: this is a NUMERICAL PRECISION tolerance, not the
        economically-derived threshold from the profitability
        analysis (t <= p*e / (n*(1-p))). Keeping this tiny is what
        makes that analysis valid: it guarantees any real,
        deliberate manipulation - not just ones exceeding some
        larger "safe" margin - is corrected with certainty whenever
        audited. If this parameter were instead set to the derived
        economic threshold, an attacker who manipulates by exactly
        that amount would be accepted-as-reported even when audited,
        making the manipulation risk-free regardless of audit
        probability (since "accepted" means the reported/fake value
        is what gets propagated). The economic threshold is computed
        separately (DFLSimulator.compute_theoretical_thresholds) as
        a theoretical reference value to compare against this
        tolerance - it is never used as a literal accept/reject
        boundary anywhere in this class.

    seed:
        Random seed for probabilistic auditing.

    Notes
    -----
    The ground-truth LCV is supplied directly by the simulator.
    This is a simulation shortcut. In the real protocol, the
    coordinator would recompute the LCV from exchanged models
    when an audit is triggered.
    """

    def __init__(
        self,
        num_clients,
        audit_probability=0.0,
        outlier_threshold=None,
        audit_threshold=1e-6,
        seed=None,
    ):

        self.num_clients = num_clients

        # -----------------------------------------------------
        # Defense parameters
        # -----------------------------------------------------

        self.audit_probability = audit_probability
        self.outlier_threshold = outlier_threshold
        self.audit_threshold = audit_threshold

        # Validate audit probability.
        if not 0.0 <= audit_probability <= 1.0:
            raise ValueError(
                "audit_probability must be between 0 and 1."
            )

        # Validate outlier threshold.
        #
        # None is explicitly allowed because it means:
        # "disable the heuristic".
        #
        if outlier_threshold is not None:

            if outlier_threshold < 0.0:
                raise ValueError(
                    "outlier_threshold must be non-negative "
                    "or None."
                )

        if audit_threshold < 0.0:
            raise ValueError(
                "audit_threshold must be non-negative."
            )

        # Independent RNG for this coordinator.
        self.rng = random.Random(seed)

        # -----------------------------------------------------
        # Contribution states
        # -----------------------------------------------------

        self.original_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }

        self.audited_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }

        # -----------------------------------------------------
        # Audit history
        # -----------------------------------------------------

        self.audit_log = []

        # -----------------------------------------------------
        # Outlier detection history
        # -----------------------------------------------------

        self.outlier_history = []

    # =========================================================
    # Contribution propagation
    # =========================================================

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

        participants = [
            client_id
        ] + list(neighbors)

        weights = network.get_weights(client_id)

        propagated = torch.zeros(
            self.num_clients
        )

        total_weight = 0.0

        for p in participants:

            w = weights[p]

            propagated += (
                w * contribution_state[p]
            )

            total_weight += w

        if total_weight > 0:

            propagated /= total_weight

        return (
            propagated
            + local_contribution_vector
        )

    # =========================================================
    # Robust contribution estimation
    # =========================================================

    def _estimate_normal_contributions(
        self,
        reported_lcv_dict,
    ):
        """
        Estimate a normal contribution for every client.

        For target client j, collect all non-zero reported
        contributions to j and use their median.

        The median is robust against a small number of malicious
        reports.

        Returns
        -------
        estimated_contributions:
            Tensor of shape [num_clients].

        values_by_target:
            Dictionary containing observed values for each
            target client.
        """

        values_by_target = {
            j: []
            for j in range(self.num_clients)
        }

        # -----------------------------------------------------
        # Collect reports.
        # -----------------------------------------------------

        for reporter_id, lcv in (
            reported_lcv_dict.items()
        ):

            for target_id in range(
                self.num_clients
            ):

                value = float(
                    lcv[target_id]
                )

                # Ignore absent sparse entries.
                if value != 0.0:

                    values_by_target[
                        target_id
                    ].append(value)

        # -----------------------------------------------------
        # Robust estimate.
        # -----------------------------------------------------

        estimated = torch.zeros(
            self.num_clients
        )

        for target_id in range(
            self.num_clients
        ):

            values = values_by_target[
                target_id
            ]

            if len(values) == 0:
                continue

            tensor_values = torch.tensor(
                values,
                dtype=torch.float32
            )

            estimated[target_id] = (
                torch.median(
                    tensor_values
                )
            )

        return (
            estimated,
            values_by_target
        )

    # =========================================================
    # Outlier detection
    # =========================================================

    def _detect_outliers(
        self,
        reported_lcv_dict,
    ):
        """
        Detect suspicious LCV reports.

        If outlier_threshold is None, the heuristic is completely
        disabled.

        In that case:

            client_outliers = False
            entry_outliers = False
            max_deviations = 0.0

        for every client.

        Otherwise, for every non-zero reported entry:

            deviation =
                |reported - estimated_normal|

        A client becomes an outlier if at least one entry exceeds
        the configured threshold.

        Returns
        -------
        client_outliers:
            {client_id: bool}

        entry_outliers:
            {
                client_id:
                    {
                        target_id: bool
                    }
            }

        estimated_contributions:
            Robust estimated contribution vector.

        max_deviations:
            Maximum deviation for every reporting client.
        """

        # =====================================================
        # None means NO HEURISTIC. Return immediately.
        # =====================================================

        if self.outlier_threshold is None:

            client_outliers = {
                client_id: False
                for client_id in reported_lcv_dict
            }

            entry_outliers = {
                client_id: {
                    target_id: False
                    for target_id in range(
                        self.num_clients
                    )
                }
                for client_id in reported_lcv_dict
            }

            max_deviations = {
                client_id: 0.0
                for client_id in reported_lcv_dict
            }

            estimated_contributions = torch.zeros(
                self.num_clients
            )

            return (
                client_outliers,
                entry_outliers,
                estimated_contributions,
                max_deviations,
            )

        # =====================================================
        # Heuristic enabled
        # =====================================================

        (
            estimated_contributions,
            values_by_target,
        ) = self._estimate_normal_contributions(
            reported_lcv_dict
        )

        client_outliers = {}
        entry_outliers = {}
        max_deviations = {}

        for reporter_id, lcv in (
            reported_lcv_dict.items()
        ):

            entry_outliers[
                reporter_id
            ] = {}

            max_deviation = 0.0
            client_is_outlier = False

            for target_id in range(
                self.num_clients
            ):

                value = float(
                    lcv[target_id]
                )

                expected = float(
                    estimated_contributions[
                        target_id
                    ]
                )

                if value == 0.0:

                    entry_outliers[
                        reporter_id
                    ][target_id] = False

                    continue

                deviation = abs(
                    value - expected
                )

                is_outlier = (
                    deviation
                    > self.outlier_threshold
                )

                entry_outliers[
                    reporter_id
                ][target_id] = is_outlier

                max_deviation = max(
                    max_deviation,
                    deviation
                )

                if is_outlier:

                    client_is_outlier = True

            client_outliers[
                reporter_id
            ] = client_is_outlier

            max_deviations[
                reporter_id
            ] = max_deviation

        return (
            client_outliers,
            entry_outliers,
            estimated_contributions,
            max_deviations,
        )

    # =========================================================
    # Probabilistic audit
    # =========================================================

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

        Both triggers are tracked separately.

        Audit reasons: None, "random", "outlier", "both".

        If audited, the reported LCV is compared against the
        deterministic ground-truth LCV using self.audit_threshold
        (a tiny numerical-precision tolerance - see class docstring
        for why this must stay small rather than being set to the
        derived economic threshold).

        A dishonest report is replaced by the ground-truth LCV,
        with the attacker's own contribution set to zero.
        """

        random_audit = (
            self.rng.random()
            < self.audit_probability
        )

        audited = (
            forced_audit
            or random_audit
        )

        if forced_audit and random_audit:

            audit_reason = "both"

        elif forced_audit:

            audit_reason = "outlier"

        elif random_audit:

            audit_reason = "random"

        else:

            audit_reason = None

        # -----------------------------------------------------
        # No audit
        # -----------------------------------------------------

        if not audited:

            return (
                reported_lcv.clone(),
                {
                    "audited": False,
                    "random_audit": False,
                    "outlier_audit": False,
                    "accepted": None,
                    "flagged": False,
                    "audit_reason": None,
                    "max_deviation": None,
                    "outlier_deviation":
                        outlier_deviation,
                },
            )

        # -----------------------------------------------------
        # Audit against ground truth
        # -----------------------------------------------------

        deviation = (
            reported_lcv
            - ground_truth_lcv
        ).abs()

        max_deviation = float(
            deviation.max()
        )

        accepted = (
            max_deviation
            <= self.audit_threshold
        )

        if accepted:

            return (
                reported_lcv.clone(),
                {
                    "audited": True,
                    "random_audit":
                        random_audit,
                    "outlier_audit":
                        forced_audit,
                    "accepted": True,
                    "flagged": False,
                    "audit_reason":
                        audit_reason,
                    "max_deviation":
                        max_deviation,
                    "outlier_deviation":
                        outlier_deviation,
                },
            )

        # -----------------------------------------------------
        # Dishonest report: replace with ground truth, zero the
        # attacker's own reward for this round.
        # -----------------------------------------------------

        corrected_lcv = (
            ground_truth_lcv.clone()
        )

        corrected_lcv[
            client_id
        ] = 0.0

        return (
            corrected_lcv,
            {
                "audited": True,
                "random_audit":
                    random_audit,
                "outlier_audit":
                    forced_audit,
                "accepted": False,
                "flagged": True,
                "audit_reason":
                    audit_reason,
                "max_deviation":
                    max_deviation,
                "outlier_deviation":
                    outlier_deviation,
            },
        )

    # =========================================================
    # Complete round
    # =========================================================

    def update_round(
        self,
        ground_truth_lcv_dict,
        reported_lcv_dict,
        network,
    ):
        """
        Update both contribution versions (original, audited).

        If outlier_threshold is None, only probabilistic auditing
        is performed. If outlier_threshold is not None, suspicious
        reports are additionally forced into the audit process.
        """

        # =====================================================
        # 1. Outlier detection
        # =====================================================

        (
            client_outliers,
            entry_outliers,
            estimated_contributions,
            max_deviations,
        ) = self._detect_outliers(
            reported_lcv_dict
        )

        num_outliers = sum(
            client_outliers.values()
        )

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
                "num_outliers":
                    num_outliers,
                "outlier_threshold":
                    self.outlier_threshold,
                "heuristic_enabled":
                    self.outlier_threshold is not None,
            }
        )

        # =====================================================
        # 2. Propagate both versions
        # =====================================================

        new_original = {}
        new_audited = {}

        round_audit_log = {}

        for client_id, reported_lcv in (
            reported_lcv_dict.items()
        ):

            ground_truth_lcv = (
                ground_truth_lcv_dict[
                    client_id
                ]
            )

            # -------------------------------------------------
            # Version 1: Original (no defense)
            # -------------------------------------------------

            new_original[
                client_id
            ] = self.update_single(
                client_id,
                reported_lcv,
                network,
                self.original_contributions,
            )

            # -------------------------------------------------
            # Version 2: Audited protocol
            # -------------------------------------------------

            forced_audit = (
                client_outliers[client_id]
            )

            outlier_deviation = (
                max_deviations[client_id]
            )

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

            new_audited[
                client_id
            ] = self.update_single(
                client_id,
                audited_lcv,
                network,
                self.audited_contributions,
            )

        # =====================================================
        # 3. Commit state
        # =====================================================

        self.original_contributions = new_original
        self.audited_contributions = new_audited

        self.audit_log.append(round_audit_log)

        print(
            "[Coordinator] Round propagation "
            "finished"
        )

    # =========================================================
    # Accessors
    # =========================================================

    def get_contribution(
        self,
        client_id,
        method="original",
    ):
        """
        Return one client's contribution vector.

        method: "original" or "audited"
        """

        mapping = {
            "original": self.original_contributions,
            "audited": self.audited_contributions,
        }

        if method not in mapping:

            raise ValueError(
                f"Unknown method: {method}"
            )

        return mapping[method][client_id]

    def get_all_contributions(self):
        """
        Return both current contribution states.
        """

        return {
            "original": self.original_contributions,
            "audited": self.audited_contributions,
        }

    def get_audit_log(self):
        """
        Return the complete audit history.
        """

        return self.audit_log

    def get_outlier_history(self):
        """
        Return the complete outlier detection history.
        """

        return self.outlier_history