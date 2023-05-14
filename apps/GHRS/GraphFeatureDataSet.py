import torch

import pandas as pd

from torch.utils.data import Dataset

class GraphFeatureDataSet(Dataset):
    def __init__(self, graphFeatureDf: pd.DataFrame):
        print(graphFeatureDf)
        self.dataFrame = graphFeatureDf

    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, index):
        row = torch.Tensor(self.dataFrame.iloc[index])
        return row
