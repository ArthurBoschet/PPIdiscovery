#object oriented
from abc import ABC, abstractmethod

#NumPy
import numpy as np

#Pytorch
import torch
from torch.utils.data import Dataset

class FastEmbedder(ABC):

  @abstractmethod
  def _embed(self, right_nodes, left_nodes):
    pass

  def __getitem__(self, edges):
    if edges.ndim > 1:
      right_nodes = edges[:,0]
      left_nodes = edges[:,1]
    else:
      right_nodes = edges[0]
      left_nodes = edges[1]
    return self._embed(right_nodes, left_nodes)

class EmbedderSingle(FastEmbedder):
  def __init__(self, keyed_vectors):
    self.keyed_vectors = keyed_vectors

class HadamardEmbedderFast(EmbedderSingle):
  def _embed(self, right_nodes, left_nodes):
    return self.keyed_vectors[right_nodes] * self.keyed_vectors[left_nodes]

class AverageEmbedderFast(EmbedderSingle):
  def _embed(self, right_nodes, left_nodes):
    return (self.keyed_vectors[right_nodes] + self.keyed_vectors[left_nodes])/2

class EmbedderList(FastEmbedder):
  def __init__(self, embedders):
    self.embedders = embedders

  def _embed(self, right_nodes, left_nodes):
    return np.hstack([embedder._embed(right_nodes, left_nodes) for embedder in self.embedders])




class EdgesDataSet(Dataset):
    def __init__(self, edges, labels, embedder):
        self.embedder = embedder
        if isinstance(edges, str) and isinstance(labels, str):
          self.edges = np.load(edges)
          self.labels = np.load(labels)
        else:
          self.edges = edges
          self.labels = labels

        print("Shuffle Dataset")
        #shuffle in unison
        self.edges, self.labels = self._shuffle_unison(self.edges, self.labels)
        #tensor of labels
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def get_subset(self, idx):
        return EdgesDataSet(self.edges[idx], self.labels[idx], self.embedder)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        edges = self.edges[idx]
        labels = self.labels[idx]
        # labels = labels.type(torch.LongTensor)
        embeddings = torch.tensor(self.embedder[edges], dtype=torch.float32)
        return embeddings, labels

    def _shuffle_unison(self, a, b):
      assert len(a) == len(b)
      p = np.random.permutation(len(a))
      return a[p], b[p]

class EdgesDataSetFast(Dataset):
    def __init__(self, edges, labels, embedder):
        self.embedder = embedder
        if isinstance(edges, str) and isinstance(labels, str):
          self.edges = np.load(edges)
          self.labels = np.load(labels)
        else:
          self.edges = edges
          self.labels = labels

        print("Shuffle Dataset")
        #shuffle in unison
        self.edges, self.labels = self._shuffle_unison(self.edges, self.labels)
        
        print("Get Embeddings")
        #tensor of labels
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        #embedded whole dataset
        self.embeddings = torch.tensor(self.embedder[self.edges], dtype=torch.float32)

    def _shuffle_unison(self, a, b):
      assert len(a) == len(b)
      p = np.random.permutation(len(a))
      return a[p], b[p]


    def get_subset(self, idx):
        return EdgesDataSet(self.edges[idx], self.labels[idx], self.embedder)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embeddings = self.embeddings[idx]
        labels = self.labels[idx]
        return embeddings, labels