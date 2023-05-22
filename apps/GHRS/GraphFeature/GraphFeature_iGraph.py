# import pandas as pd
# import igraph as ig

# from tqdm import tqdm

# import apps.GHRS.GraphFeature.GraphFeature as GraphFeature

# from apps.GHRS.GraphFeature.GraphFeature import TimeTaken

# class GraphFeature_iGraph(GraphFeature.GraphFeature):
#   def _addGraphEdges(self) -> None:
#     self.G = ig.Graph()
#     self.G.add_vertices(len(self.users_df))

#     for el in tqdm(self.edge_list, desc='_getGraph::add_edge', total=1655185):
#       self.G.add_edge(el[0], el[1])
#       self.G.add_edge(el[0], el[0])
#       self.G.add_edge(el[1], el[1])

#   @TimeTaken
#   def _getGraphFeatures(self) -> pd.DataFrame:
#     return super()._getGraphFeatures()
  
#   @TimeTaken
#   def _calcPagerank(self) -> None:
#     pr = self.G.pagerank()
#     self.graphFeature2DataFrame('PR', pr)
#   @TimeTaken
#   def _calcDegreeCentrality(self) -> None:
#     dc = self.G.degree()
#     self.graphFeature2DataFrame('DC', dc)
#     ...
#   @TimeTaken
#   def _calcClosenessCentrality(self) -> None:
#     cc = self.G.clossness()
#     self.graphFeature2DataFrame('CC', cc)
#     ...
#   @TimeTaken
#   def _calcBetweennessCentrality(self) -> None:
#     bc = self.G.betweenness()
#     self.graphFeature2DataFrame('BC', bc)
#     ...