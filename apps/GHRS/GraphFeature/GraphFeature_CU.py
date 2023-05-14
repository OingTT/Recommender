import cudf, cugraph

from tqdm import tqdm

import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken

class GraphFeature_CU(GraphFeature.GraphFeature):
  def _getEdgeList(self) -> None:
    self.G = cugraph.Graph()

    for el in tqdm(self.edge_list, desc='_getGraph::add_edge', total=1655185):
      self.G.add_edge(el[0], el[1], weight=1)
      self.G.add_edge(el[0], el[0], weight=1)
      self.G.add_edge(el[1], el[1], weight=1)

    self.users_cudf = cudf.DataFrame(self.users_df)
    self.cuG = cugraph.from_networkx(self.G)
  
  @TimeTaken
  def _calcPagerank(self) -> None:
    pr = cugraph.pagerank(self.G)
    self.graphFeature2DataFrame('PR', pr)

  @TimeTaken
  def _calcDegreeCentrality(self) -> None:
    dc = cugraph.degree_centrality(self.G)
    self.graphFeature2DataFrame('CD', dc)

  @TimeTaken
  def _calcClosenessCentrality(self) -> None:
    # cc = cugraph.closeness_centrality(self.G)
    # self.graphFeature2DataFrame('CC', cc)
    ...

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = cugraph.betweenness_centrality(self.G)
    self.graphFeature2DataFrame('CB', bc)

  @TimeTaken
  def _calcLoadCentrality(self) -> None:
    # lc = cugraph.load_centrality(self.G)
    # self.graphFeature2DataFrame('LC', lc)
    ...
    
  @TimeTaken
  def _calcAverageNeighborDegree(self) -> None:
    # nd = cugraph.average_neighbor_degree(self.G, weight='weight')
    # self.graphFeature2DataFrame('AND', nd)
    ...
