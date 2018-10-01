import re
import numpy as np 
from collections import Counter
import MeCab
import itertools
from igraph import *
from aozora import Aozora 

minfreq = 4
m = MeCab.Tagger("-Ochasen")

def readin(filename):
	with open(filename, "r") as afile:
		whole_str = afile.read()
	sentenses = (re.sub('。', '。\n', whole_str)).splitlines()
	return [re.sub(' ', '', u) for u in sentenses if len(u) != 0]

filename = "abe.txt"
string = readin(filename)

sentensemeishilist = [
	[v.split()[2] for v in m.parse(sentense).splitlines()
		if (len(v.split()) >= 3 and v.split()[3][:2]=='名詞')]
		for sentense in string]

doubletslist = [
	list(itertools.combinations(meishilist,2))
		for meishilist in sentensemeishilist if len(meishilist) >= 2]
alldoublets = []
for u in doubletslist:
	alldoublets.extend(u)

dcnt = Counter(alldoublets)

#print('pair frequency', sorted(dcnt.items(), key=lambda x: x[1], reverse=True))

restricteddict = dict(((k, dcnt[k]) for k in dcnt.keys() if dcnt[k] >= minfreq))
charedges = restricteddict.keys()
vertices = list(set([v[0] for v in charedges] + [v[1] for v in charedges]))

edges = [(vertices.index(u[0]), vertices.index(u[1])) for u in charedges]
g = Graph(vertex_attrs={"label": vertices, "name": vertices}, edges=edges, directed=False)
plot(g, vertex_size=30, bbox=(1000,1000), vertex_color='white')
