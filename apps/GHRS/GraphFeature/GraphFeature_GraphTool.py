from graph_tool.all import *
from graph_tool import centrality

from tqdm import tqdm
from typing import Dict
import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

from apps.GHRS.GraphFeature.GraphFeature import TimeTaken

class GraphFeature_GraphTool(GraphFeature.GraphFeature):
  def _addGraphEdges(self, edge_list: list) -> None:
    self.graph = Graph()
    
    for el in tqdm(edge_list, desc='_getGraph::add_edge'):
      e = self.graph.add_edge(el[0], el[0])
      e = self.graph.add_edge(el[0], el[1])
      e = self.graph.add_edge(el[1], el[0])
    
  def VPM2dict(self, vpm: VertexPropertyMap) -> Dict:
    '''
    Convert Graph-tool's VertexPropertyMap to Dictionary
    '''
    converted = dict()
    for v in self.graph.vertices():
      converted.update({
        int(v): float(vpm[int(v)])
      })
      
    return converted

  def BC2dict(self, BC: tuple) -> Dict:
    '''
    Convert Graph-tool's Betweenness Centrality to Dictionary
    '''
    converted = dict()
    for v in self.graph.vertices():
      converted.update({
        int(v): float(BC[0][int(v)])
      })
    return converted
  
  def EC2dict(self, EC: tuple[VertexPropertyMap]) -> Dict:
    '''
    Convert Graph-tool's EigenValue Centrality to Dictionary
    '''
    converted = dict()
    for v in self.graph.vertices():
      converted.update({
        int(v): float(EC[1][int(v)])
      })
    return converted
  
  @TimeTaken
  def _calcPagerank(self):
    '''
    PageRank: Metric for measuring the importance of nodes in a graph
    '''
    pr = centrality.pagerank(self.graph)
    pr_dict = self.VPM2dict(pr)
    self.concatGraphFeatureToUsers('PR', pr_dict)
    return pr

  @TimeTaken
  def _calcBetweennessCentrality(self):
    '''
    Betweenness Centrality: Metric for measuring the centrality of a node within a graph based on shortest paths
    '''
    bc = centrality.betweenness(self.graph)
    bc_dict = self.BC2dict(bc)
    self.concatGraphFeatureToUsers('BC', bc_dict)
    return bc

  @TimeTaken
  def _calcClosenessCentrality(self):
    '''
    Closeness Centrality: Metric for measuring the centrality of a node in network
    '''
    cc = centrality.closeness(self.graph)
    cc_dict = self.VPM2dict(cc)
    self.concatGraphFeatureToUsers('CC', cc_dict)
    return cc

  @TimeTaken
  def _calcEigenvectorCentrality(self):
    '''
    Eigenvector Centrality: Metric for measuring the influence of a node in a network
    '''
    ec = centrality.eigenvector(self.graph)
    ec_dict = self.EC2dict(ec)
    self.concatGraphFeatureToUsers('EC', ec_dict)
    return ec
  
  @TimeTaken
  def _calcKatzCentrality(self):
    '''
    Katz Centrality: Metric for measuring the centrality of a node within a graph based on the centrality of its neighbors
    '''
    kz = centrality.katz(self.graph)
    kz_dict = self.VPM2dict(kz)
    self.concatGraphFeatureToUsers('KZ', kz_dict)
    return kz