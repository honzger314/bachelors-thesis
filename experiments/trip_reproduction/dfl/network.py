import random
import networkx as nx

class Network:
    """
    Communication graph for DFL.

    Stores which clients can exchange models.

    Example ring with 5 clients:

        0 ---- 1
        |      |
        4 ---- 2
              |
              3


    Each client communicates only with its neighbors.
    """

    def __init__(
        self,
        num_clients,
        topology="ring"
    ):

        self.num_clients = num_clients

        self.topology = topology

        self.edges = self.create_topology()

    def get_weights(self, client_id):
        """
        Returns w_j^(i,t) for j in N(i,t) = {client_id} ∪ neighbors(client_id).

        Currently uniform. This is the ONLY place to change later
        for non-uniform weighting (e.g. by dataset size) — every
        other file reads weights through this method.
        """

        participants = [client_id] + list(self.neighbors(client_id))

        weight = 1.0 / len(participants)

        return {p: weight for p in participants}

    def create_topology(self):
        """
        Creates the communication graph.

        Returns:

        {
            client_id: [
                neighbor_ids
            ]
        }

        """

        if self.topology == "ring":

            return self.create_ring()


        elif self.topology == "fully_connected":

            return self.create_fully_connected()


        elif self.topology == "line":

            return self.create_line()

        elif self.topology == "star":

            return self.create_star()

        elif self.topology == "random":

            return self.create_random()

        elif self.topology == "watts_strogatz":

            return self.create_watts_strogatz()

        else:
            raise ValueError(
                f"Unknown topology: {self.topology}"
            )

    def create_watts_strogatz(self, degree=4, rewiring_probability=0.3):
        """
        Watts-Strogatz small-world topology.

        degree:
            Number of neighbors per client before rewiring.
            Must be even.

        rewiring_probability:
            Probability of rewiring each edge.
        """

        if degree >= self.num_clients:
            raise ValueError(
                "Watts-Strogatz degree must be smaller than num_clients."
            )

        if degree % 2 != 0:
            raise ValueError(
                "Watts-Strogatz degree must be even."
            )

        graph = nx.watts_strogatz_graph(
            n=self.num_clients,
            k=degree,
            p=rewiring_probability,
            seed=random.randint(0, 2**32 - 1),
        )

        return {
            i: list(graph.neighbors(i))
            for i in range(self.num_clients)
        }

    def create_ring(self):
        """
        Ring topology.

        Example:

        0: [1,4]
        1: [0,2]
        2: [1,3]
        3: [2,4]
        4: [3,0]

        """

        edges = {}


        for i in range(self.num_clients):

            left = (
                i - 1
            ) % self.num_clients


            right = (
                i + 1
            ) % self.num_clients


            edges[i] = [
                left,
                right
            ]


        return edges



    def create_line(self):
        """
        Line topology.

        Example:

        0 -- 1 -- 2 -- 3 -- 4


        """

        edges = {}


        for i in range(self.num_clients):

            neighbors = []


            if i > 0:
                neighbors.append(
                    i - 1
                )


            if i < self.num_clients - 1:
                neighbors.append(
                    i + 1
                )


            edges[i] = neighbors


        return edges



    def create_fully_connected(self):
        """
        Every client communicates with every other client.

        Mostly useful as a debugging baseline.
        """

        edges = {}


        for i in range(self.num_clients):

            edges[i] = [
                j
                for j in range(self.num_clients)
                if j != i
            ]


        return edges

    def create_star(self):

        edges = {}

        center = 0

        for i in range(self.num_clients):

            if i == center:
                edges[i] = [
                    j for j in range(self.num_clients)
                    if j != center
                ]

            else:
                edges[i] = [
                    center
                ]

        return edges

    def create_random(self, probability=0.3):

        edges = {
            i: []
            for i in range(self.num_clients)
        }


        for i in range(self.num_clients):

            for j in range(i+1, self.num_clients):

                if random.random() < probability:

                    edges[i].append(j)
                    edges[j].append(i)


        return edges

    def neighbors(
        self,
        client_id
    ):
        """
        Returns neighbors of a client.
        """

        return self.edges[client_id]



    def print_network(self):
        """
        Debug helper.
        """

        print(
            f"Topology: {self.topology}"
        )

        for client, neighbors in self.edges.items():

            print(
                f"Client {client}: {neighbors}"
            )