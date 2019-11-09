import numpy as np
import copy
from itertools import chain, combinations

class Network:

    def __init__(self, phi):
        self.phi = phi

    def compute_effort(self, G, alpha):
        n, m = G.shape
        return np.linalg.inv(np.eye(n) - self.phi * G) @ alpha

    def compute_payoff(self, G, C, alpha):
        n, m = G.shape
        x = self.compute_effort(G, alpha)
        bool_matrix = G == 1
        cost = np.array([sum(C[i, bool_matrix[i]]) for i in range(n)])
        return 1/2 * x**2 - cost

    def powerset(self, iterable):
        s = list(iterable)
        if s == []:
            return s
        else:
            return chain.from_iterable(combinations(s,r) for r in range(1, len(s)+1))

    def check_best_response(self, G, C, alpha):
        n, m = G.shape
        NB = {}
        payoff = self.compute_payoff(G, C, alpha)
        for i in range(n):
            max_payoff = payoff[i]
            remove_edges = []
            edges_index = [j for j, x in enumerate(G[i,:]) if x == 1]
            delete_edges = self.powerset(edges_index)
            for delete_edge in delete_edges:
                G_copy = copy.copy(G)
                for edge in delete_edge:
                    G_copy[i, edge] = 0
                new_payoff = self.compute_payoff(G_copy, C, alpha)
                if new_payoff[i] > max_payoff:
                    max_payoff = new_payoff[i]
                    remove_edges = list(delete_edge)
            if remove_edges == []:
                NB[i] = []
            else:
                NB[i] = remove_edges
        
        return NB

    def main(self, G, C, alpha, verbose=True):
        """
        G : potential network
        """
        NB = self.check_best_response(G, C, alpha)
        NB_agents = [k for k, v in NB.items() if v != []]
        while NB_agents != []:
            agent = np.random.choice(NB_agents, 1)
            remove_edges = NB[int(agent)]
            for edge in remove_edges:
                G[agent, edge] = 0
            if verbose:
                print("agent {0} removes {1}".format(agent, remove_edges))
            NB = self.check_best_response(G, C, alpha)
            NB_agents = [k for k, v in NB.items() if v != []]
        return G

    def find_key_player(self, G, C, alpha):
        """
        simple algorithm by definition
        there may be much room to improve

        G : potential network
        """
        n, m = G.shape
        importances = []
        eqm_G = self.main(G, C, alpha, verbose=False)
        total_efforts = sum(self.compute_effort(eqm_G, alpha)) 
        for i in range(n):
            deleted_G = np.delete(np.delete(G,i,0),i,1)
            deleted_C = np.delete(np.delete(C,i,0),i,1)
            deleted_alpha = np.delete(alpha, i)
            deleted_eqm_G = self.main(deleted_G, deleted_C, deleted_alpha, verbose=False)
            deleted_total_efforts = sum(self.compute_effort(deleted_eqm_G, deleted_alpha))
            importances.append(total_efforts - deleted_total_efforts)
        key_player = np.where(importances == np.max(importances))
        return key_player


def find_key_player_fix(G, alpha, phi):
    """
    find a key player in the fixed network
    this method is for comparing my model with previous ones

    G : fixed network
    """
    n, m = G.shape
    importances = []
    M = np.linalg.inv(np.eye(n) - phi * G)
    b = M @ alpha
    for i in range(n):
        x = b[i] * sum(M[:,i]) / M[i,i]
        importances.append(x)
    key_player = np.where(importances == np.max(importances))
    return key_player




if __name__ == "__main__":
    n = 5 # number of agents
    phi = 1/n

    G = np.ones((n,n))
    for i in range(n):
        G[i, i] = 0


"""
    ## Example of Large Discontinuity in Network

    alpha = np.ones(n)

    epsilon = 0.05

    C = np.array([[0, 3 - epsilon, 3 - epsilon, 3 - epsilon, 3 - epsilon],
                  [3 - epsilon, 0, 3 - epsilon, 3 - epsilon, 3 - epsilon],
                  [3 - epsilon, 3 - epsilon, 0, 3 - epsilon, 3 - epsilon],
                  [3 - epsilon, 3 - epsilon, 3 - epsilon, 0, 3 - epsilon],
                  [3 - epsilon, 3 - epsilon, 3 - epsilon, 3 - epsilon, 0]])

    algo = Network(phi)
    eqm_G = algo.main(G, C, alpha, verbose=False)
    print("=====Before=====")
    print("efforts : {0}".format(algo.compute_effort(eqm_G, alpha)))
    print("payoffs : {0}".format(algo.compute_payoff(eqm_G, C, alpha)))
    print("key player : {0}".format(algo.find_key_player(G, C, alpha)))
    print(eqm_G)

    C_hat = np.array([[0, 3 + 4 * epsilon, 3 - epsilon, 3 - epsilon, 3 - epsilon],
                      [3 - epsilon, 0, 3 - epsilon, 3 - epsilon, 3 - epsilon],
                      [3 - epsilon, 3 - epsilon, 0, 3 - epsilon, 3 - epsilon],
                      [3 - epsilon, 3 - epsilon, 3 - epsilon, 0, 3 - epsilon],
                      [3 - epsilon, 3 - epsilon, 3 - epsilon, 3 - epsilon, 0]])

    algo = Network(phi)
    eqm_G = algo.main(G, C_hat, alpha, verbose=False)
    print("=====After=====")
    print("efforts : {0}".format(algo.compute_effort(eqm_G, alpha)))
    print("payoffs : {0}".format(algo.compute_payoff(eqm_G, C_hat, alpha)))
    print("key player : {0}".format(algo.find_key_player(G, C_hat, alpha)))
    print(eqm_G)
"""

    ## Example of Comparison with My Model and Previous Models in Finding Key Players

alpha = np.ones(n)

C = np.array([[0, 3.6, 0.2, 0.2, 0.2],
              [0.3, 0, 0.2, 4, 4.5],
              [0.2, 0.2, 0, 4.5, 4.3],
              [4.1, 0.2, 0.4, 0, 6.5],
              [3.2, 4.1, 0.3, 1, 0]])

G_fix = np.array([[0, 0, 1, 1, 1],
                  [1, 0, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0]])

algo = Network(phi)
eqm_G = algo.main(G, C, alpha, verbose=False)
key_player = algo.find_key_player(G, C, alpha)
key_player_fix = find_key_player_fix(G_fix, alpha, phi)

print("=====Endogenous Network=====")
print(eqm_G)
print("key player : {0}".format(key_player))
print("=====Fixed Network=====")
print("key player : {0}".format(key_player_fix))
 