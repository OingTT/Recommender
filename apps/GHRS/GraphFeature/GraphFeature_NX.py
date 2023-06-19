import pandas as pd
import networkx as nx

from tqdm import tqdm

import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken

class GraphFeature_NX(GraphFeature.GraphFeature):
  def _addGraphEdges(self) -> None:
    self.G = nx.Graph()

    for el in tqdm(self.edge_list, desc='_getGraph::add_edge', total=1655185):
      self.G.add_edge(el[0], el[1], weight=1)
      self.G.add_edge(el[0], el[0], weight=1)
      self.G.add_edge(el[1], el[1], weight=1)

  @TimeTaken
  def _calcPagerank(self) -> None:
    pr = nx.pagerank(self.G.to_directed())
    self.concatGraphFeatureToUsers('PR', pr)

  @TimeTaken
  def _calcDegreeCentrality(self) -> None:
    dc = nx.degree_centrality(self.G)
    self.concatGraphFeatureToUsers('DC', dc)
  @TimeTaken
  def _calcClosenessCentrality(self) -> None:
    cc = nx.closeness_centrality(self.G)
    self.concatGraphFeatureToUsers('CC', cc)

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = nx.betweenness_centrality(self.G)
    self.concatGraphFeatureToUsers('BC', bc)

  @TimeTaken
  def _calcLoadCentrality(self) -> None:
    lc = nx.load_centrality(self.G)
    self.concatGraphFeatureToUsers('LC', lc)

  @TimeTaken
  def _calcAverageNeighborDegree(self) -> None:
    nd = nx.average_neighbor_degree(self.G, weight='weight')
    self.concatGraphFeatureToUsers('AND', nd)
