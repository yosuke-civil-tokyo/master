# VAE model to generate models' adjacency matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepGenerativeModel(nn.Module):
    def __init__(self, in_dim=)