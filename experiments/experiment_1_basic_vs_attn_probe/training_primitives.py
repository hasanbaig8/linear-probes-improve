from enum import Enum
import torch.nn as nn
import torch

class DataSplit(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"

class LogisticClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
    
    def forward(self,x: torch.Tensor):
        return self.lin1(x)