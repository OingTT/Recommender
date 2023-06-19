import cudf, cugraph

import pandas as pd
import networkx as nx

from tqdm import tqdm

import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken

class GraphFeature_RAPIDS(GraphFeature.GraphFeature):
  def _addGraphEdges(self) -> None:
    self.G = nx.Graph()

    for el in tqdm(self.edge_list, desc='_getGraph::add_edge', total=1655185):
      self.G.add_edge(el[0], el[1], weight=1)
      self.G.add_edge(el[0], el[0], weight=1)
      self.G.add_edge(el[1], el[1], weight=1)

  @TimeTaken
  def _calcPagerank(self) -> None:
    pr = cugraph.pagerank(self.G)
    self.concatGraphFeatureToUsers('PR', pr)

  @TimeTaken
  def _calcDegreeCentrality(self) -> None:
    dc = cugraph.degree_centrality(self.G)
    self.concatGraphFeatureToUsers('DC', dc)

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = cugraph.betweenness_centrality(self.G)
    self.concatGraphFeatureToUsers('CB', bc)

  @TimeTaken
  def _calcEigenvectorCentrality(self) -> None:
    ec = cugraph.centrality.eigenvector_centrality(self.G)
    self.concatGraphFeatureToUsers('EC', ec)

