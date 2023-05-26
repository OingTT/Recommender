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
  def _getGraphFeatures(self) -> pd.DataFrame:
    self._calcPagerank()
    self._calcDegreeCentrality()
    self._calcBetweennessCentrality()
    self._calcEigenvectorCentrality()
    graphFeature_df = self.users_df[self.users_df.columns[0:]]
    graphFeature_df.fillna(0, inplace=True)
    return graphFeature_df
  
  @TimeTaken
  def _calcPagerank(self) -> None:
    pr = cugraph.pagerank(self.G)
    self.graphFeature2DataFrame('PR', pr)

  @TimeTaken
  def _calcDegreeCentrality(self) -> None:
    dc = cugraph.degree_centrality(self.G)
    self.graphFeature2DataFrame('DC', dc)

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = cugraph.betweenness_centrality(self.G)
    self.graphFeature2DataFrame('CB', bc)

  @TimeTaken
  def _calcEigenvectorCentrality(self) -> None:
    ec = cugraph.centrality.eigenvector_centrality(self.G)
    self.graphFeature2DataFrame('EC', ec)

