import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

#######################################################################
################################ Model ################################
#######################################################################

class PoseFFNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()

        hidden_dims = range(input_dim, output_dim, 4)[1:]

        # Define the layers of the network
        # TODO - add drop out!
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_dims[2], output_dim)

    #######################################################################
    
    def forward(self, featureVector):

        output = None
        output = self.linear1(featureVector)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output = self.linear3(output)
        output = self.relu3(output)
        output = self.linear4(output)

        return output
    
    #######################################################################