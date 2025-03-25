import torch
import torch.nn as nn
import torch.nn.functional as F 

class Discriminator(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x


def build_discriminator(args):
    return Discriminator(args.input_dim, args.hidden_dim, args.output_dim)