import networkx as nx
import numpy as np
import itertools
import random
from copy import deepcopy

def density(g):
  return 2*len(g.edges)/(len(g.nodes)**2)


def sample_negative_edges(G, num_neg):
  negs = set()
  proteins = G.nodes
  edges = set(G.edges)
  while len(negs) < num_neg:
    p1, p2 = random.sample(proteins, 2)
    if (p1, p2) not in edges and (p2, p1) not in edges:
      negs.add((p1, p2))
  return list(negs)

def realistic_dataset_maker(G, excluded_percent=0.05, network_density=None, G_train=None):

  if G_train is None:
    #excluded edges
    excluded_edges = random.sample(G.edges, int(len(G.edges)*excluded_percent))
    G_train = deepcopy(G)
    G_train.remove_edges_from(excluded_edges)
  else:
    G_excluded = deepcopy(G)
    G_excluded.remove_edges_from(G_train.edges)
    excluded_edges = list(G_excluded.edges)

  assert len(G_train.edges) < len(G.edges)

  #needed number of negative samples
  if network_density is None:
    network_density = density(G)
  negative_sampling_number = int(len(excluded_edges)*(1/network_density - 1))

  #sampled negative edges
  negative_egdes = sample_negative_edges(G, negative_sampling_number)

  #dataset
  X = np.vstack([np.array([e[0], e[1]]) for e in negative_egdes + excluded_edges])
  Y = [0 for _ in range(len(negative_egdes))] + [1 for _ in range(len(excluded_edges))]

  print(f"The density is {np.sum(Y)/len(Y)}")

  return X, Y, G_train