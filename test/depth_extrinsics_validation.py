import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import flying3d_io
import ipdb 
st = ipdb.set_trace
import pydisco_utils 

rgb1 = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/RGB_cleanpass/left/0006.png"
rgb2 = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/RGB_cleanpass/right/0006.png"
camera_file = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/camera_data.txt"
disp_f = "/Users/shamitlal/Desktop/shamit/cmu/katefgroup/raft_datasets/Sampler/FlyingThings3D/disparity/0006.pfm"

left_extr = "-0.735639512539 0.094669058919 -0.670725047588 -0.0747194886205 -0.653832554817 0.159533917904 0.739629507065 3.7231169343 0.177023410797 0.982642769814 -0.0554617233574 11.0321881771 0.0 0.0 0.0 1.0"
right_extr = "-0.735639512539 0.094669058919 -0.670725047588 -0.810359001159 -0.653832554817 0.159533917904 0.739629507065 3.06928437948 0.177023410797 0.982642769814 -0.0554617233574 11.2092115879 0.0 0.0 0.0 1.0"

origin_T_camLeft = torch.tensor(flying3d_io.get_origin_T_camX_from_string(left_extr)).unsqueeze(0)
origin_T_camRight = torch.tensor(flying3d_io.get_origin_T_camX_from_string(right_extr)).unsqueeze(0)
camRight_T_camLeft = pydisco_utils.safe_inverse(origin_T_camRight) @ origin_T_camLeft

rgb1 = flying3d_io.read(rgb1)
rgb1_ = rgb1.reshape(-1,3)
rgb2 = flying3d_io.read(rgb2)

disparity = flying3d_io.readPFM(disp_f)[0]
st()
disparity = np.ascontiguousarray(disparity, dtype=np.float32)
disparity = torch.tensor(disparity).unsqueeze(0)
baseline = torch.tensor([1])
pix_T_camX = torch.tensor(pydisco_utils.get_pix_T_camX("things"))

focallen = pix_T_camX[0, 0, 0].reshape(1)
depth = pydisco_utils.disp2depth(baseline, focallen, disparity)
xyz_camLeft = pydisco_utils.depth2pointcloud(depth.unsqueeze(0), pix_T_camX)
xyz_camRight = pydisco_utils.apply_4x4(camRight_T_camLeft.float(), xyz_camLeft.float())
out = np.zeros_like(rgb1)

projected_camRight = pydisco_utils.apply_pix_T_cam(pix_T_camX, xyz_camRight)

projected_camRight[:,:,0] = torch.clamp(projected_camRight[:,:,0], 0, rgb1.shape[0]-1)
projected_camRight[:,:,1] = torch.clamp(projected_camRight[:,:,1], 0, rgb1.shape[1]-1)
projected_camRight = projected_camRight.int()
out[projected_camRight[0,:,1], projected_camRight[0,:,0], :] = rgb1_
vis = np.concatenate([out, rgb1, rgb2], axis=0)
plt.imshow(vis)
plt.show(block=True)
