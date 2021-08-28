import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Thin Plate Spline Transformation (TPS)
class transformer_TSP(nn.Module):
    
    def __init__(self, image_batch, image_size, image_rectified_size, image_channel_number = 1):
        
        super(transformer_TSP, self).__init__()
        
        self.image_batch = image_batch
        self.image_size = image_size
        self.image_rectified_size = image_rectified_size
        self.image_channel_number = image_channel_number
        self.localization = self.localization_network(self.image_batch, self.image_channel_number)
        self.generator = self.grid_generator(self.image_batch, self.image_rectified_size)
        
        
    def forward(self, image_batch):
        fiducial_points = self.localization_network(image_batch)
        
        
        
        

# Localization Network of RARE
class localization_network(nn.Module):
    
    def __init__(self, image_batch, image_channel_number = 1):
        

# Grid Generator of RARE
class grid_generator(nn.Module):
    
    def __init__(self, image_batch, image_rectified_size):
        
        
        
        
        
