import graph_tool as gt
from graph_tool.all import *
from graph_tool import centrality

from tqdm import tqdm

import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken


class GraphFeature_GraphTool(GraphFeature.GraphFeature):
  def _addGraphEdges(self) -> None:
    self.G = Graph()

    for el in tqdm(self.edge_list, desc='_getGraph::add_edge'):
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
  def _calcPagerank(self) -> None:
    pr = pagerank(self.G)
    pr_dict = self.VPM2dict(pr)
    self.graphFeature2DataFrame('PR', pr_dict)

  @TimeTaken
  def _calcDegreeCentrality(self) -> None:
    # dc = centrality.degree(self.G)
    # self.graphFeature2DataFrame('DC', dc)
    ...

  @TimeTaken
  def _calcClosenessCentrality(self) -> None:
    cc = centrality.closeness(self.G)
    cc_dict = self.VPM2dict(cc)
    self.graphFeature2DataFrame('CC', cc_dict)

  @TimeTaken
  def _calcBetweennessCentrality(self) -> None:
    bc = centrality.betweenness(self.G)
    bc_dict = self.BC2dict(bc)
    self.graphFeature2DataFrame('BC', bc_dict)

  @TimeTaken
  def _calcLoadCentrality(self) -> None:
    # lc = centrality.load(self.G)
    # self.graphFeature2DataFrame('LC', lc)
    ...

  @TimeTaken
  def _calcEigenVectorCentrality(self) -> None:
    ec = centrality.eigenvector(self.G)
    self.graphFeature2DataFrame('EC', ec)

  @TimeTaken
  def _calcAverageNeighborDegree(self) -> None:
    # nd = centrality.average_neightbor_degree(self.G)
    # self.graphFeature2DataFrame('ND', nd)
    ...
    