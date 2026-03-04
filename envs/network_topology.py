
import random
import networkx as nx

def create_network_graph(num_nodes):
        """
        Create a scalable and connected graph representing the USANET topology.
        """
        g = nx.Graph()
        # Add nodes with hardware attributes
        for i in range(num_nodes):
                g.add_node(i, 
                        cpu=f'{random.uniform(2.0, 4.0):.2f} GHz',
                        ram=f'{random.randint(50,120)} GB',
                        storage=f'{random.randint(100, 1000)} GB',
                        energy=f'{random.randint(20,50)}'
                )

            # Define a backbone structure with 30 interconnected nodes
        backbone_nodes = min(30, num_nodes)
        base_edges = [
                (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
                (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 11),
                (9, 11), (10, 11), (7, 8), (9, 10), (4, 5), (8, 9),
                (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                (3, 6), (2, 4), (1, 9), (8, 10), (7, 9), (10, 12), (11, 13),
                (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19),
                (18, 20), (19, 21), (20, 22), (21, 23), (22, 24), (23, 25),
                (24, 26), (25, 27), (26, 28), (27, 29), (28, 0), (29, 1)
            ]

            # Add backbone edges
        g.add_edges_from(base_edges)
        # Dynamically expand the network if num_nodes > 30
        if num_nodes > backbone_nodes:
                for i in range(backbone_nodes, num_nodes):
                    # Connect new nodes to backbone nodes for strong connectivity
                    backbone_node = random.choice(range(backbone_nodes))
                    g.add_edge(i, backbone_node)

                    # Additional random connections for better redundancy
                    for _ in range(random.randint(1, 3)):
                        another_node = random.choice(range(num_nodes))
                        if another_node != i:
                            g.add_edge(i, another_node)

            # Assign bandwidth and latency to edges
        for u, v in g.edges():
                g[u][v]['bandwidth'] = 0
                g[u][v]['latency'] = 0  # Adjusted range for better visualization

        return g