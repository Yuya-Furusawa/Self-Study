import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 

n = 100000
roop = 10
weight_cut = list(map(lambda w: w * 0.05, range(20)))

#Generate a scale-free graph with random weights
def nx_ba_1(n, ww):
	G = nx.barabasi_albert_graph(n, 5)
	edges = [e for e in G.edges()]
	weights = np.random.random_sample((len(edges),))

	G_A = nx.Graph()
	G_A.add_nodes_from(range(n))

	for (i,edge) in enumerate(edges):
		G_A.add_edges_from([edge], weight=weights[i])

	#remove the edges which doesn't attain some level of weight
	for (u, v, d) in G_A.copy().edges(data=True):
		if d["weight"] <= ww:
			G_A.remove_edge(u, v)

	#Compute a component size and return the size of giant component
	con =[len(c) for c in sorted(nx.connected_components(G_A), key=len, reverse=True)]
	return con[0]

#Compute average of the giant component size
def ave_percent(n, ww):
	now = 0
	sum_ = 0
	while now < roop:
		sum_ += nx_ba_1(n, ww)
		now += 1
	ave = sum_ / roop
	ave_percent = ave / n * 100
	return ave_percent

#Return ave_percent for each weight
def graph(n):
	graphs = []
	for w in weight_cut:
		graphs.append(ave_percent(n, w))
	return graphs

#Draw a Graph
plt.plot(weight_cut, graph(n), marker='o')
plt.grid(True)
plt.plot()
plt.show()