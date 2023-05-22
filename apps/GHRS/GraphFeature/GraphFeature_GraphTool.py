import pandas as pd
from graph_tool.all import *
from graph_tool import centrality

from tqdm import tqdm

import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken


class GraphFeature_GraphTool(GraphFeature.GraphFeature):
  def _addGraphEdges(self) -> None:
    self.G = Graph()

    for el in tqdm(self.edge_list, desc='_getGraph::add_edge', total=1655185):
      self.G.add_edge_list([
        (el[0], el[0]),
        (el[0], el[1]),
        (el[1], el[0])
      ])

  def VPM2dict(self, vpm: VertexPropertyMap) -> dict:
    '''
    Convert Graph-tool's VertexPropertyMap to Dictionary
    '''
    converted = dict()
    for v in self.G.vertices():
      converted.update({
        int(v): float(vpm[int(v)])
      })
    return converted

  def BC2dict(self, BC: tuple) -> dict:
    '''
    Convert Graph-tool's Betweenness Centrality to Dictionary
    '''
    converted = dict()
    for v in self.G.vertices():
      converted.update({
        int(v): float(BC[0][int(v)])
      })
    return converted
  
  @TimeTaken
  def _getGraphFeatures(self) -> pd.DataFrame:
    self._calcPagerank()
    self._calcBetweennessCentrality()
    self._calcClosenessCentrality()
    self._calcEigenvectorCentrality()
    graphFeature_df = self.users_df[self.users_df.columns[0:]]
    graphFeature_df.fillna(0, inplace=True)
    return graphFeature_df
  
  @TimeTaken
  def _calcPagerank(self) -> None:
    pr = pagerank(self.G)
    pr_dict = self.VPM2dict(pr)
    self.graphFeature2DataFrame('PR', pr_dict)

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = centrality.betweenness(self.G)
    bc_dict = self.BC2dict(bc)
    self.graphFeature2DataFrame('BC', bc_dict)

  @TimeTaken
  def _calcClosenessCentrality(self) -> None:
    cc = centrality.closeness(self.G)
    cc_dict = self.VPM2dict(cc)
    self.graphFeature2DataFrame('CC', cc_dict)

  @TimeTaken
  def _calcEigenvectorCentrality(self) -> None:
    ec = centrality.eigenvector(self.G)
    ec_dict = self.VPM2dict(ec)
    self.graphFeature2DataFrame('EC', ec_dict)
