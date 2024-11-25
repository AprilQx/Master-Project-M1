#this is the model file for the neural network model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#First version
class Mnist_Addition_nn(nn.Module):
    def  _init_(self, hidden_size=128,num_layers=2,dropout_rate=0.2):
        super().__init__()
        layers=[nn.Flatten(),
                nn.linear(1568,hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)]
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        layers.append(nn.Linear(hidden_size, 19))
        self.network=nn.Sequential(*layers)
        self.embedding=layers[-2]
    def forward(self, x):
        return self.network(x)
