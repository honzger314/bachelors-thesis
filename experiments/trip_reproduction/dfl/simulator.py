import torch

from dfl.client import Client
from dfl.network import Network
from trip.coordinator import Coordinator

from data.dataset import create_client_loaders
from utils.model_utils import evaluate_model


class DFLSimulator:
    """
    Decentralized Federated Learning simulator.

    There is no central server.
    Clients communicate only through
    the defined network topology.
    """

    def __init__(
        self,
        num_clients=5,
        rounds=20,
        local_epochs=1,
        batch_size=64,
        topology="ring",
        device="cpu"
    ):

        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device


        # Create datasets
        client_loaders, test_loader = create_client_loaders(
            num_clients=num_clients,
            batch_size=batch_size
        )

        self.test_loader = test_loader


        # Create clients
        self.clients = []

        for i in range(num_clients):

            client = Client(
                client_id=i,
                train_loader=client_loaders[i],
                device=device
            )

            self.clients.append(client)


        # Create communication graph
        self.network = Network(
            num_clients=num_clients,
            topology=topology
        )


        # TRIP-Shapley coordinator
        self.coordinator = Coordinator(
            num_clients=num_clients
        )


    def train_round(self, round_number):
        """
        Executes one TRIP-Shapley DFL round.

        Order:

        1. Local training
        2. Exchange pre/post models
        3. Compute LCVs
        4. Send LCVs to coordinator
        5. Aggregate models
        """


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

            neighbor_ids = self.network.neighbors(
                client.id
            )


            received = []


            # Own model
            received.append(
                client.create_message()
            )


            # Neighbor models
            for n in neighbor_ids:

                received.append(
                    self.clients[n].create_message()
                )


            messages[client.id] = received


            print(
                f"Client {client.id} receives "
                f"{len(received)} models"
            )



        #
        # 3. Compute LCVs
        #

        print("\n--- Computing Local Contribution Vectors ---")


        lcv_dict = {}


        for client in self.clients:

            print(
                f"\n[Client {client.id}] "
                "Starting LCV"
            )


            lcv = client.compute_lcv(
                received_messages=messages[client.id],
                test_loader=self.test_loader
            )


            print(
                f"[Client {client.id}] "
                "Finished LCV"
            )


            #
            # Dictionary -> vector
            #

            vector = torch.zeros(
                self.num_clients
            )


            for cid, value in lcv.items():

                vector[cid] = value


            lcv_dict[client.id] = vector



        #
        # 4. Coordinator update
        #

        print("\n--- Updating coordinator ---")


        self.coordinator.update_round(
            lcv_dict,
            self.network
        )


        print(
            "Coordinator update complete"
        )



        #
        # 5. Aggregate models
        #

        print("\n--- Model aggregation ---")


        for client in self.clients:

            print(
                f"Aggregating client {client.id}"
            )

            weights = self.network.get_weights(client.id)

            client.aggregate(
                received_messages=messages[client.id],
                weights=weights
            )


        print(
            f"Round {round_number + 1} complete"
        )



    def evaluate(self):
        """
        Evaluate all client models
        on the shared test set.
        """

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
        """
        Run complete DFL training.
        """

        print(
            "Starting DFL training"
        )


        self.network.print_network()


        for r in range(self.rounds):

            self.train_round(
                round_number=r
            )


            print(
                "\n--- Evaluation ---"
            )


            accuracies = self.evaluate()


            mean_accuracy = sum(
                accuracies
            ) / len(accuracies)


            print(
                f"Round {r+1}/{self.rounds} "
                f"| Mean accuracy: "
                f"{mean_accuracy:.4f}"
            )


        print(
            "Training finished"
        )


        return self.clients