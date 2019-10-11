import numpy as np
import copy
from itertools import chain, combinations

class Network:

    def __init__(self, phi, n, C, alpha):
        self.n = n
        self.phi = phi
        self.C = C
        self.alpha = alpha

    def compute_payoff(self, G):
        x = np.linalg.inv(np.eye(self.n) - self.phi * G) @ self.alpha
        bool_matrix = G == 1
        cost = np.array([sum(self.C[i, bool_matrix[i]]) for i in range(self.n)])
        return 1/2 * x**2 - cost

    def powerset(self, iterable):
        s = list(iterable)
        if s == []:
            return s
        else:
            return chain.from_iterable(combinations(s,r) for r in range(1, len(s)+1))

    def check_best_response(self, G):
        NB = {}
        payoff = self.compute_payoff(G)
        print(payoff)
        for i in range(self.n):
            max_payoff = payoff[i]
            remove_edges = []
            edges_index = [j for j, x in enumerate(G[i,:]) if x == 1]
            delete_edges = self.powerset(edges_index)
            for delete_edge in delete_edges:
                G_copy = copy.copy(G)
                for edge in delete_edge:
                    G_copy[i, edge] = 0
                new_payoff = self.compute_payoff(G_copy)
                print("agent {0}, delete_edge {1}, new_payoff {2}".format(i, delete_edge, new_payoff))
                if new_payoff[i] > max_payoff:
                    max_payoff = new_payoff[i]
                    remove_edges = list(delete_edge)
                elif new_payoff[i] == max_payoff:
                    if remove_edges == []:
                        remove_edges.append(edges_index)
                    remove_edges.append(list(delete_edge))
            NB[i] = np.random.choice(remove_edges, 1)
        
        return NB

    def main(self, G):
        NB = self.check_best_response(G)
        NB_agents = [k for k, v in NB.items() if v != []]
        while NB_agents != []:
            agent = np.random.choice(NB_agents, 1)
            remove_edges = NB[int(agent)]
            for edge in remove_edges:
                G[agent, edge] = 0
            print("agent {0} removes {1}".format(agent, remove_edges))
            NB = self.check_best_response(G)
            NB_agents = [k for k, v in NB.items() if v != []]
        return G


if __name__ == "__main__":
    n = 3 # number of agents
    phi = 1/3

    G = np.ones((n,n))
    for i in range(n):
        G[i, i] = 0 

    alpha = np.ones(n)

    # cost matrix
    """
    C = np.array([[0, 6, 0.3, 3, 2],
                  [0.2, 0, 1, 0.02, 1],
                  [0.01, 3.5, 0, 2, 0],
                  [0.2, 7, 0.1, 0, 0.1],
                  [0.5, 1, 0.12, 0.2, 0]])
    """
    C = np.array([[0, 1.7, 1.7],
                  [1.3, 0, 0.1],
                  [0.5, 2, 0]])
    algo = Network(phi, n, C, alpha)

    print(algo.main(G))