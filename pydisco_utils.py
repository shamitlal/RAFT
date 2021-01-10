import numpy as np 
import torch 
import ipdb 
# import open3d as o3d 
st = ipdb.set_trace 
import torch.nn.functional as F

device = torch.device('cuda')

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K
    
def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd

def sceneflow_intrinsics_to_pydisco(intrinsics):
    pix_T_camXs = eye_4x4(intrinsics.shape[0])
    pix_T_camXs[:,0,0] = intrinsics[:, 0]
    pix_T_camXs[:,1,1] = intrinsics[:, 1]
    pix_T_camXs[:,0,2] = intrinsics[:, 2]
    pix_T_camXs[:,1,2] = intrinsics[:, 3]
    return pix_T_camXs

def visualize_pcd(pts, rgb=None):
    if rgb != None:
        pts = torch.cat([pts, rgb], dim=-1)
    pcd = make_pcd(pts)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame, pcd])

def get_pix_T_camX(dataset_name, B=1):
    if dataset_name == "things":
        pix_T_camXs =  torch.tensor([[1050.0, 0.0, 479.5, 0],
                            [0.0, 1050.0, 269.5, 0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]).to(device)
    else:
        assert False, "Invalid dataset"
    
    return pix_T_camXs.unsqueeze(0).repeat(B, 1, 1)

def disp2depth(baseline, focallen, disparity):
    '''
    baseline = B
    focallen = B
    disparity = BHW
    '''
    baseline = baseline.reshape(-1,1,1)
    focallen = focallen.reshape(-1,1,1)
    depth = (baseline*focallen)/disparity
    return depth.unsqueeze(1)


def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = meshgrid2d(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def meshgrid2d(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
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

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

# def get_flow_field(depth, translation, pix_T_camX):
#     # TODO: after baseline, change translation to T
#     B, _, H, W = depth.shape
#     xyz_camXS = depth2pointcloud(depth, pix_T_camX)
#     proj_xyS = apply_pix_T_cam(pix_T_camX, xyz_camXS)

#     xyz_camXT = xyz_camXS + translation.reshape(B, H*W, 3)
#     proj_xyT = apply_pix_T_cam(pix_T_camX, xyz_camXT)
#     flow_field = proj_xyT - proj_xyS

#     return flow_field.reshape(B,H,W,-1), proj_xyT.reshape(B,H,W,-1)

def get_flow_field(depth, translation, pix_T_camX):

    B, _, H, W = depth.shape

    xyz_camXS = depth2pointcloud(depth, pix_T_camX)
    proj_xyS = apply_pix_T_cam(pix_T_camX, xyz_camXS)
    
    B, N, _= xyz_camXS.shape
    xyz_camXS_ = xyz_camXS.reshape(B*N, 1, 3)

    translation_ = translation.reshape(B*H*W, 1, 3)
    xyz_camXT_ = xyz_camXS_ + translation_
    xyz_camXT = xyz_camXT_.reshape(B, N, 3)

    depth_camXT = xyz_camXT[:,:,2].reshape(B,1,H,W)
    

    proj_xyT = apply_pix_T_cam(pix_T_camX, xyz_camXT)
    cond = depth_camXT>0.01
    cond_ = cond.reshape(B, H*W, 1).repeat(1,1,2)
    proj_xyT = torch.where(cond_, proj_xyT, proj_xyT.detach())

    depth_camXT = torch.where(cond, depth_camXT, depth_camXT.detach())
    inv_depth_camXT = 1./(depth_camXT + 1e-5)

    flow_field = proj_xyT - proj_xyS

    return flow_field.reshape(B,H,W,-1), proj_xyT.reshape(B,H,W,-1), inv_depth_camXT

    
def create_depth_image_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    xy = torch.round(xy).long()
    depth = torch.zeros(H*W, dtype=torch.float32, device=device)
    
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid]
    z = z[valid]

    inds = sub2ind(H, W, xy[:,1], xy[:,0]).long()
    depth[inds] = z
    valid = (depth > 0.0).float()
    depth[torch.where(depth == 0.0)] = 1000.0
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def sub2ind(height, width, y, x):
    return y*width + x

def create_depth_image(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=device)
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=device)
    for b in list(range(B)):
        depth[b], valid[b] = create_depth_image_single(xy[b], z[b], H, W)
    return depth, valid

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def eye_4x4(B):
    rt = torch.eye(4, device=device).view(1,4,4).repeat([B, 1, 1])
    return rt

def eye_3x3(B):
    rt = torch.eye(3, device=device).view(1,3,3).repeat([B, 1, 1])
    return rt

def grid_sample(input, grid, isnormalized=False):
    B, _, H, W = input.shape
    if not isnormalized:
        grid[:,:,:,0]  = 2*grid[:,:,:,0]/(W-1) - 1.0
        grid[:,:,:,1]  = 2*grid[:,:,:,1]/(H-1) - 1.0

    out = F.grid_sample(input, grid, align_corners=True)
    return out 

