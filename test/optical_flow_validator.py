import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import flying3d_io
import ipdb 
st = ipdb.set_trace

rgb1 = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/RGB_cleanpass/left/0006.png"
rgb2 = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/RGB_cleanpass/left/0007.png"
flow1 = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/optical_flow/forward/0006.pfm"

rgb1 = flying3d_io.read(rgb1)
rgb2 = flying3d_io.read(rgb2)
flow1 = flying3d_io.read(flow1)

rgbs = np.concatenate([rgb1, rgb2], axis=0)
plt.imshow(rgbs)
plt.show(block=True)
# plt.imshow(rgb2)
# plt.show(block=True)
out = np.zeros_like(rgb1)
for i in range(rgb1.shape[0]):
    for j in range(rgb1.shape[1]):
        desti = int(flow1[i, j, 1] + i)
        destj = int(flow1[i, j, 0] + j)
        try:
            out[desti, destj] = rgb1[i,j]
        except:
            pass

out1 = np.concatenate([out, rgb1, rgb2], axis=0)
plt.imshow(out1)
plt.show(block=True)
plt.imshow(flow1)
plt.show(block=True)
st()
aa=1