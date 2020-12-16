import torch 
import numpy as np 
import matplotlib.pyplot as plt
import pytorch3d
import ipdb 
st = ipdb.set_trace 
from liegroups.torch import SE3, SO3, utils

# returns w_hat
def hat(w):
    # w -> Bx3
    # out -> Bx3x3
    B = w.shape[0]
    zz = torch.zeros(B)
    v1, v2, v3 = w[:, 0], w[:, 1], w[:, 2]
    w_hat = torch.zeros((B, 3, 3)).to(w.device)
    w_hat[:, 0, 1] = -v3
    w_hat[:, 0, 2] = v2
    w_hat[:, 1, 0] = v3
    w_hat[:, 1, 2] = -v1 
    w_hat[:, 2, 0] = -v2
    w_hat[:, 2, 1] = v1 

    return w_hat

def logmap(T):
    
    shape = T.shape
    assert shape[-1]==4
    assert shape[-1]==4
    T = T.reshape(-1, 4, 4)
    twist = SE3.log(SE3.from_matrix(T.cpu())).to(T.device)
    outshape = list(shape)[:-2] + [6]
    twist = twist.reshape(outshape)
    return twist

def expmap(twist):
    # Bx6xHxW
    twist = twist.permute(0, 2, 3, 1)
    B, H, W, C = twist.shape
    assert C==6
    twist = twist.reshape(B*H*W, C)
    pose = SE3.exp(twist.cpu()).as_matrix().to(twist.device)
    pose = pose.reshape(B, H, W, 4, 4)
    return pose 

if __name__ == '__main__':
    w = torch.tensor([[1,2,3]]).repeat(2,1)
    out = hat(w)
    print(out.shape)
    print(out)
