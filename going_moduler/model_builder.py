"""
Contains Pytorch model code to instantiate a TinyVGG model.
"""

import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = hidden_units * 16 * 16,  # multiplying with shape of input images after conv_block_2
                out_features = output_shape)
            )
        
        
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        
        return x
        
        ## return self.classifier(self.conv_block_2(self.conv_block_1(x)))  # benifits from operator fusion. 
