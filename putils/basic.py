import torch
import numpy as np
import torch.nn.functional as F

def gridcloud2d(B, Y, X, norm=False):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy


def meshgrid2d(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def normalize_gridcloud2d(xy, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xy[...,0]
    y = xy[...,1]
    
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xy = torch.stack([x,y], dim=-1)
    
    if clamp_extreme:
        xy = torch.clamp(xy, min=-2.0, max=2.0)
    return xy