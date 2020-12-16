import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb 
st = ipdb.set_trace
import pydisco_utils
import se3_utils 
class DenseSE3(nn.Module):
    def __init__(self):
        super(DenseSE3, self).__init__()

    def calculate_affinity(self, embeddings):

        B, C, H, W = embeddings.shape
        embeddings = embeddings.reshape(B, C, H*W).permute(0, 2, 1)
        embeddings1 = embeddings.unsqueeze(2)
        embeddings2 = embeddings.unsqueeze(1)

        diff = embeddings1 - embeddings2 
        norm = diff.norm(2, dim=-1)
        affinity = torch.exp(-norm)
        return affinity

    def get_pie_jacobian(self, X, pix_T_camX, d_dash):
        # X -> BxNx3
        # pix_T_camX -> Bx4x4
        # d_dash -> BxN
        B, N, _ = X.shape
        X = X.reshape(B*N, 3)
        d_dash = d_dash.reshape(B*N)
        pix_T_camX = pix_T_camX.unsqueeze(1).repeat(1,N,1,1).reshape(B*N, 4, 4)

        fx, fy, x0, y0 = pydisco_utils.split_intrinsics(pix_T_camX)
        a00 = fx*d_dash 
        a02 = -fx*X[:,0]*d_dash**2
        a11 = fy*d_dash
        a12 = -fy*X[:,1]*d_dash**2
        a22 = -d_dash**2

        out = torch.zeros((B*N, 3, 3)).to(X.device)
        out[:, 0, 0] = a00  
        out[:, 0, 2] = a02  
        out[:, 1, 1] = a11  
        out[:, 1, 2] = a12 
        out[:, 2, 2] = a22 

        out = out.reshape(B, N, 3, 3)
        return out

    def get_transformation_jacobian(self, X):
        # X -> BxNx3
        B, N, _ = X.shape 
        X = X.reshape(B*N, 3)
        I = pydisco_utils.eye_3x3(B*N)
        X_hat = se3_utils.hat(X)
        out = torch.cat([I, X_hat], dim=-1)
        return out.reshape(B, N, 3, 6)

    def forward(self, embeddings, revisions, weights, depth, pix_T_camXs, Tmat):
        affinity = self.calculate_affinity(embeddings)
        B, _, H, W = depth.shape
        affinity = affinity.reshape(affinity.shape[0], H, W, -1)
        xyz_camX = pydisco_utils.depth2pointcloud(depth, pix_T_camXs)
        delta_twists = torch.zeros((B, H, W, 6)).to(depth.device)

        for b in range(embeddings.shape[0]):

            pix_T_camXs_b = pix_T_camXs[b:b+1]
            weights_b = weights[b:b+1]
            for i in range(embeddings.shape[1]):
                for j in range(embeddings.shape[2]):
                    
                    T_i = Tmat[b:b+1, i, j]
                    T_j = Tmat[b].reshape(-1, 4, 4)
                    X_j = xyz_camX[b].unsqueeze(1)
                    affinity_j = affinity[b:b+1, i, j]                        
                    affinity_j = affinity_j.reshape(affinity_j.shape[0], H, W)

                    TjXj = pydisco_utils.apply_4x4(T_j, X_j)
                    TiXj = pydisco_utils.apply_4x4(T_i, X_j)
                    pie_TjXj = pydisco_utils.pie(TjXj, pix_T_camXs_b.repeat(TjXj.shape[0], 1, 1), H, W)[:, 0]
                    pie_TiXj = pydisco_utils.pie(TiXj, pix_T_camXs_b.repeat(TiXj.shape[0], 1, 1), H, W)[:, 0]
                    r_ = revisions.permute(0,2,3,1)[b].reshape(-1, 3) # H*W, 3

                    residual = pie_TiXj - (pie_TjXj + r_)

                    # Create jacobians
                    J_pie = self.get_pie_jacobian(TiXj.permute(1,0,2), pix_T_camXs_b, pie_TiXj[:,2].unsqueeze(0))
                    J_T = self.get_transformation_jacobian(X_j.permute(1,0,2))

                    Jacobian = J_pie @ J_T 
                    Jacobian = Jacobian.reshape(-1, H*W*3, 6)

                    wts = affinity_j.unsqueeze(1).repeat(1,3,1,1)*weights_b
                    wts = wts.permute(0, 2, 3, 1).reshape(-1, H*W*3)
                    wts = wts[0]
                    Jacobian = Jacobian[0]

                    J_T = Jacobian.T * wts.reshape(1,-1)
                    H_ = J_T @ Jacobian
                    rhs = Jacobian.T @ residual.reshape(-1, 1)
                    delta_ijb = torch.inverse(H_) @ rhs
                    delta_twists[b,i,j] = delta_ijb[:, 0]

        delta_transformation = se3_utils.expmap(delta_twists.permute(0,3,1,2))
        updated_transformation = delta_transformation @ Tmat
        return updated_transformation
