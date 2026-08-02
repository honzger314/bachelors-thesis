import torch
import copy


class Coordinator:
    """
    TRIP-Shapley coordinator maintaining two contribution versions.

    Original:
        Uses received LCV vectors unchanged.

    Modified:
        Removes self-contribution:
            ψ_i(i,t) = 0

    The coordinator:
        - does NOT see model parameters
        - does NOT train models
        - only receives LCVs
        - tracks contribution propagation
    """

    def __init__(
        self,
        num_clients
    ):

        self.num_clients = num_clients


        #
        # φ_i^(0)=0
        #
        # Original TRIP-Shapley state
        #
        self.original_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }


        #
        # Modified method state
        #
        self.modified_contributions = {
            i: torch.zeros(num_clients)
            for i in range(num_clients)
        }



    def update_single(
        self,
        client_id,
        local_contribution_vector,
        network,
        contribution_state
    ):
        """
        Performs one coordinator update.

        This is shared by original and modified versions.
        """

        neighbors = network.neighbors(
            client_id
        )


        participants = [
            client_id
        ] + list(neighbors)


        weights = network.get_weights(
            client_id
        )


        propagated = torch.zeros(
            self.num_clients
        )


        total_weight = 0.0


        for p in participants:

            w = weights[p]

            propagated += (
                w *
                contribution_state[p]
            )

            total_weight += w



        if total_weight > 0:

            propagated /= total_weight



        return (
            propagated +
            local_contribution_vector
        )



    def update_round(
        self,
        lcv_dict,
        network
    ):
        """
        Update both contribution versions.

        lcv_dict:

        {
            client_id:
                ψ(i,t)
        }

        """

        print(
            f"[Coordinator] Updating contributions for "
            f"{len(lcv_dict)} clients"
        )


        #
        # Temporary dictionaries because all
        # clients update simultaneously.
        #

        new_original = {}
        new_modified = {}



        for client_id, lcv in lcv_dict.items():

            print(
                f"[Coordinator] Updating client {client_id}"
            )


            #
            # Version 1:
            # Original TRIP-Shapley
            #

            new_original[client_id] = self.update_single(
                client_id,
                lcv,
                network,
                self.original_contributions
            )



            #
            # Version 2:
            # Modified method
            #

            modified_lcv = copy.deepcopy(
                lcv
            )


            #
            # Remove self contribution
            #

            modified_lcv[client_id] = 0.0



            new_modified[client_id] = self.update_single(
                client_id,
                modified_lcv,
                network,
                self.modified_contributions
            )


            print(
                "[Coordinator] Round propagation finished"
            )



        #
        # Synchronize updates
        #

        self.original_contributions = new_original

        self.modified_contributions = new_modified



    def get_contribution(
        self,
        client_id,
        method="original"
    ):
        """
        Return one client's contribution vector.
        """

        if method == "original":

            return self.original_contributions[client_id]


        elif method == "modified":

            return self.modified_contributions[client_id]


        else:

            raise ValueError(
                f"Unknown method: {method}"
            )



    def get_all_contributions(
        self
    ):
        """
        Return both contribution histories.
        """

        return {
            "original": self.original_contributions,
            "modified": self.modified_contributions
        }