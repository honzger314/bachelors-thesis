import random

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
        topology="ring",
        average_degree=4,
        rewire_prob=0.1,
        random_seed=None
    ):

        self.num_clients = num_clients

        self.topology = topology

        # Used by watts_strogatz and dense_random
        self.average_degree = average_degree
        self.rewire_prob = rewire_prob
        self.rng = random.Random(random_seed)

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

        elif self.topology == "dense_random":

            return self.create_dense_random()

        else:
            raise ValueError(
                f"Unknown topology: {self.topology}"
            )



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

        WARNING: with the current exact-Shapley LCV computation,
        a fully connected graph on n clients gives each client
        n participants (self + n-1 neighbors), which makes
        compute_shapley enumerate up to 2^n unique coalitions
        per client, per round. For n=10 that's 1024 coalitions
        per client (10,240 per round), each requiring a full
        model averaging + test-set evaluation. This scales as
        O(2^n) and becomes very expensive quickly - use
        create_dense_random or create_watts_strogatz with a
        bounded degree instead if you need a dense but tractable
        comparison topology.
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

    def create_watts_strogatz(self):
        """
        Watts-Strogatz small-world topology.

        Construction:
            1. Start from a ring lattice where each node connects
               to its k nearest neighbors (k/2 on each side),
               k = self.average_degree (must be even).
            2. Rewire each edge with probability self.rewire_prob,
               replacing one endpoint with a uniformly random node
               (no self-loops, no duplicate edges).

        Average degree stays k both before and after rewiring,
        since rewiring only relocates edges, it never adds/removes
        them.
        """

        k = self.average_degree

        if k % 2 != 0:
            raise ValueError(
                "average_degree must be even for watts_strogatz "
                f"(got {k})"
            )

        if k >= self.num_clients:
            raise ValueError(
                "average_degree must be smaller than num_clients "
                f"(got degree={k}, num_clients={self.num_clients})"
            )

        n = self.num_clients

        # Use sets during construction to make dedup / no-self-loop
        # checks easy, convert to sorted lists at the end.
        edge_set = set()

        # Step 1: ring lattice with k/2 neighbors on each side
        for i in range(n):
            for offset in range(1, k // 2 + 1):
                j = (i + offset) % n
                edge_set.add(tuple(sorted((i, j))))

        edges_list = list(edge_set)

        # Step 2: rewire
        for idx, (a, b) in enumerate(edges_list):

            if self.rng.random() < self.rewire_prob:

                # Try to rewire endpoint b to a new random node,
                # avoiding self-loops and existing edges.
                attempts = 0
                while attempts < 100:

                    new_b = self.rng.randrange(n)

                    candidate = tuple(sorted((a, new_b)))

                    if (
                        new_b != a
                        and candidate not in edge_set
                    ):
                        edge_set.discard((a, b) if a < b else (b, a))
                        edge_set.add(candidate)
                        edges_list[idx] = candidate
                        break

                    attempts += 1

        # Build adjacency dict from final edge set
        edges = {i: [] for i in range(n)}

        for a, b in edge_set:
            edges[a].append(b)
            edges[b].append(a)

        for i in range(n):
            edges[i] = sorted(edges[i])

        return edges

    def create_dense_random(self):
        """
        Bounded-degree dense random graph: a tractable alternative
        to fully_connected. Targets self.average_degree by wiring
        each node to that many random distinct neighbors (roughly
        - final degrees vary a bit due to the symmetric wiring,
        similar to an Erdos-Renyi graph tuned to the target degree).

        Unlike fully_connected (degree n-1), this keeps the number
        of Shapley coalitions per client bounded and predictable:
        with average_degree=k, players per client ~= k+1, so unique
        coalitions ~= 2^(k+1) instead of 2^n.
        """

        n = self.num_clients

        # probability chosen so expected degree ~= average_degree
        # for an Erdos-Renyi-style graph: p = k / (n - 1)
        probability = self.average_degree / (n - 1)

        edges = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < probability:
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

        degrees = [len(v) for v in self.edges.values()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        print(
            f"Average degree: {avg_degree:.2f} "
            f"(min={min(degrees)}, max={max(degrees)})"
        )