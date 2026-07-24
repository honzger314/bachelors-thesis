import torch


class Coordinator:
    """
    Lightweight TRIP-Shapley coordinator.

    The coordinator:
        - does NOT see model parameters
        - does NOT train models
        - only receives LCVs
        - tracks contribution propagation

    Implements Equation (3):

        φ_i^(t+1)
        =
        weighted inherited contribution
        +
        local contribution vector
    """

    def __init__(
        self,
        num_clients
    ):

        self.num_clients = num_clients


        #
        # φ_i^(0)=0
        #
        # Every client starts with no
        # known contribution.
        #

        self.contributions = {

            i: torch.zeros(
                num_clients
            )

            for i in range(num_clients)

        }



    def update(
        self,
        client_id,
        neighbor_ids,
        local_contribution_vector,
        weights=None
    ):
        """
        Update contribution vector
        of one client.

        Parameters
        ----------
        client_id:

            Client whose model is updated.


        neighbor_ids:

            N(i,t) excluding itself.


        local_contribution_vector:

            ψ(i,t)


        weights:

            Aggregation weights w_ij


        """


        #
        # Include the client itself
        #
        # N(i,t)=neighbors + {i}
        #

        participants = [
            client_id
        ] + list(neighbor_ids)



        if weights is None:

            weights = {
                p: 1.0 / len(participants)
                for p in participants
            }



        #
        # First term of Equation (3)
        #
        # propagated previous influence
        #

        propagated = torch.zeros(
            self.num_clients
        )


        total_weight = 0.0


        for p in participants:

            weight = weights[p]


            propagated += (
                weight
                *
                self.contributions[p]
            )


            total_weight += weight



        if total_weight > 0:

            propagated /= total_weight



        #
        # Second term:
        #
        # new contribution from this round
        #

        new_vector = (
            propagated
            +
            local_contribution_vector
        )


        self.contributions[client_id] = new_vector



    def update_round(
        self,
        lcv_dict,
        network
    ):
        """
        Update all clients for one round.

        Parameters:

        lcv_dict:

            {
              client_id:
                  ψ(i,t)
            }

        network:

            DFL network object

        """
        print(
            f"[Coordinator] Updating contributions for "
            f"{len(lcv_dict)} clients"
        )

        new_values = {}


        for client_id, lcv in lcv_dict.items():
            print(
                f"[Coordinator] Updating client {client_id}"
            )
            neighbors = network.neighbors(
                client_id
            )


            participants = [
                client_id
            ] + neighbors


            weights = network.get_weights(client_id)


            propagated = torch.zeros(
                self.num_clients
            )

            total_weight = 0.0


            for p in participants:

                w = weights[p]

                propagated += (
                    w
                    *
                    self.contributions[p]
                )

                total_weight += w


            if total_weight > 0:
                propagated /= total_weight


            new_values[client_id] = (
                propagated
                +
                lcv
            )

            print(
                "[Coordinator] Round propagation finished"
            )



        #
        # Synchronize all updates after
        # the round.
        #
        # Important because all clients
        # update simultaneously.
        #

        self.contributions = new_values



    def get_contribution(
        self,
        client_id
    ):
        """
        Return final φ_i.
        """

        return self.contributions[client_id]



    def get_all_contributions(self):
        """
        Return all contribution vectors.
        """

        return self.contributions