import flying3d_io
import pydisco_utils
import numpy as np 
import torch 
import matplotlib.pyplot as plt
import ipdb 
st = ipdb.set_trace

device = torch.device('cpu')

disp_f = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/disparity/0006.pfm"
rgb_f = "/Users/shamitlal/Downloads/Sampler/FlyingThings3D/RGB_cleanpass/left/0006.png"
disparity = flying3d_io.readPFM(disp_f)[0]
st()
rgb = torch.tensor(flying3d_io.read(rgb_f)).float()
disparity = np.ascontiguousarray(disparity, dtype=np.float32)
disparity = torch.tensor(disparity).unsqueeze(0)
baseline = torch.tensor([1])
pix_T_camX = torch.tensor(pydisco_utils.get_pix_T_camX()).unsqueeze(0).to(device)
focallen = pix_T_camX[0, 0, 0].reshape(1)
depth = pydisco_utils.disp2depth(baseline, focallen, disparity)
xyz_camX = pydisco_utils.depth2pointcloud(depth.unsqueeze(1), pix_T_camX)
pydisco_utils.visualize_pcd(xyz_camX[0], rgb.reshape(-1, 3))
