import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb 
st = ipdb.set_trace
import denseSE3
import pydisco_utils
import se3_utils 
from torch.autograd import gradcheck


def checkgrad():
    B, H, W = 1, 2, 3
    pix_T_camXs = torch.tensor([[[98.4375,  0.0000, 44.9531,  0.0000],
            [ 0.0000, 77.7778, 19.9630,  0.0000],
            [ 0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.0000]]]).cuda().repeat(B,1,1).double()
    denseSE3Func = DenseSE3Func.apply
    embeddings = torch.randn((B, 32, H, W),dtype=torch.double,requires_grad=True).cuda().double()
    affinity = denseSE3.DenseSE3.calculate_affinity(embeddings).double()
    affinity = affinity.reshape(affinity.shape[0], H, W, H, W).double()
    revisions = torch.randn((B, 3, H, W),dtype=torch.double,requires_grad=True).cuda().double()
    weights = torch.randn((B, 3, H, W),dtype=torch.double,requires_grad=True).cuda().double()
    depth = torch.randn((B, 1, H, W),dtype=torch.double,requires_grad=True).cuda().double()
    Tmat = torch.randn((B, H, W, 4, 4),dtype=torch.double,requires_grad=True).cuda().double()
    xyz_camX = pydisco_utils.depth2pointcloud(depth, pix_T_camXs).double()
    xyz_camX = xyz_camX.reshape(B, H, W, 3).double()
    test = gradcheck(denseSE3Func, (revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat))
    print(test)
    # out = denseSE3Func(revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat)
    # loss = torch.sum(out*out)
    # loss.backward()
    # back = DenseSE3Func.backward(out)

def apply_denseSE3_func(embeddings, revisions, weights, depth, pix_T_camXs, Tmat):
    denseSE3Func = DenseSE3Func.apply
    affinity = denseSE3.DenseSE3.calculate_affinity(embeddings)
    B, _, H, W = depth.shape
    affinity = affinity.reshape(affinity.shape[0], H, W, H, W)
    xyz_camX = pydisco_utils.depth2pointcloud(depth, pix_T_camXs)
    xyz_camX = xyz_camX.reshape(B, H, W, 3)
    denseSE3Func(revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat)


class DenseSE3Func(torch.autograd.Function):

    @staticmethod
    def applyRT(RT, xyz):
        #RT -> 3x4
        #xyz -> 3
        R = RT[:3,:3]
        t = RT[:3, -1]

        return torch.matmul(R, xyz) + t

    @staticmethod
    def applyRT_backward(D_out, RT, xyz):
        #D_xyz -> 3
        R = RT[:3,:3]
        t = RT[:3, -1]
        D_t = D_out.clone()
        D_R = torch.matmul(D_out.unsqueeze(1), xyz.unsqueeze(0))
        D_xyz = torch.matmul(D_out.T, R).T 

        D_RT = torch.zeros_like(RT).to(RT.device).double()
        D_RT[:3, :3] = D_R
        D_RT[:3, -1] = D_t

        return D_RT, D_xyz

    @staticmethod
    def apply_pix_T_camX(K, xyz):
        fx = K[0,0]
        fy = K[1,1]
        x0 = K[0,2]
        y0 = K[1,2]
        x,y,z = xyz 
        # EPS = 1e-4
        z = z + 1e-5
        # z = torch.clamp(z, min=EPS)
        x = (x*fx)/(z)+x0
        y = (y*fy)/(z)+y0
        return torch.stack([x, y])

    @staticmethod
    def apply_pix_T_camX_backward(D_xy, K, xyz):
        # d_xy -> 2
        fx = K[0,0]
        fy = K[1,1]
        x0 = K[0,2]
        y0 = K[1,2]

        x, y, z = xyz
        # EPS = 1e-4
        # z = torch.clamp(z, min=EPS)
        z = z + 1e-5

        D_K = torch.zeros((4, 4)).to(D_xy.device).double()
        D_K[0, 2] = D_xy[0]
        D_K[1, 2] = D_xy[1]
        D_K[0, 0] = D_xy[0]*x/z
        D_K[1, 1] = D_xy[1]*y/z
        D_z = -D_xy[0]*(x*fx/(z*z)) - D_xy[1]*(y*fy/(z*z))
        D_x = D_xy[0]*fx/z
        D_y = D_xy[1]*fy/z
        
        D_xyz = torch.stack([D_x, D_y, D_z])
        return D_K, D_xyz

    @staticmethod
    def pie(K, xyz):
        xy = DenseSE3Func.apply_pix_T_camX(K, xyz)
        x, y, z = xyz
        z = z.reshape(1)
        z = z + 1e-5
        zinv = 1./(z)
        out = torch.cat([xy, zinv], dim=-1)
        return out

    @staticmethod
    def pie_backward(D_out, K, xyz):
        # D_out -> 3
        x, y, z = xyz
        D_xy = D_out[:2]
        D_zinv = D_out[-1]
        D_K, D_xyz = DenseSE3Func.apply_pix_T_camX_backward(D_xy, K, xyz)
        z = z + 1e-5
        D_z = -D_zinv/(z)**2
        D_xyz[-1] += D_z

        return D_K, D_xyz
        
    @staticmethod
    def pie_jacobian(xyz, K, d_dash):
        X, Y, _ = xyz
        fx = K[0,0]
        fy = K[1,1]
        out = torch.zeros(3, 3).to(K.device).double()
        out[0, 0] = fx*d_dash
        out[0, 2] = -fx*X*(d_dash**2)
        out[1, 1] = fy*d_dash
        out[1, 2] = -fy*Y*(d_dash**2)
        out[2, 2] = -d_dash**2
        return out

    @staticmethod
    def pie_jacobian_backward(D_out, xyz, K, d_dash):
        X, Y, _ = xyz
        fx = K[0,0]
        fy = K[1,1]

        D_K = torch.zeros_like(K).to(K.device).double()
        D_fx = D_out[0, 0]*d_dash + D_out[0, 2]*(-X*(d_dash**2))
        D_fy = D_out[1, 1]*d_dash + D_out[1, 2]*(-Y*(d_dash**2))
        D_K[0, 0] = D_fx
        D_K[1, 1] = D_fy

        D_X = D_out[0, 2]*(-fx*(d_dash**2))
        D_Y = D_out[1, 2]*(-fy*(d_dash**2))
        D_Z = 0 
        D_xyz = torch.tensor([D_X, D_Y, D_Z]).to(D_out.device)

        D_dash = D_out[0, 0]*fx + D_out[1, 1]*fy + D_out[2, 2]*(-2*d_dash) + D_out[0, 2]*(-2*fx*X*d_dash) + D_out[1, 2]*(-2*fy*Y*d_dash)        
        return D_xyz, D_K, D_dash

    @staticmethod
    def transformation_jacobian(xyz):
        x,y,z = xyz
        out = torch.zeros((3, 6)).to(xyz.device).double()
        out[0, 0] = 1
        out[1, 1] = 1
        out[2, 2] = 1
        
        out[0, 4] = -z 
        out[0, 5] = y
        out[1, 3] = z
        out[1, 5] = -x
        out[2, 3] = -y
        out[2, 4] = x
        return out

    @staticmethod
    def transformation_jacobian_backward(D_out):
        D_X = -D_out[1, 5] + D_out[2, 4]
        D_Y = -D_out[2, 3] + D_out[0, 5]
        D_Z = -D_out[0, 4] + D_out[1, 3]
        
        D_xyz = torch.stack([D_X, D_Y, D_Z])
        return D_xyz

    @staticmethod
    def jacobian_stuff(J_pie, J_Trans, residual, w_final):
        Jacobian = torch.matmul(J_pie, J_Trans)
        Jacobian_w = Jacobian*w_final.unsqueeze(1)

        H_inter = torch.matmul(Jacobian.T, Jacobian_w)
        b_inter = torch.matmul(Jacobian_w.T, residual)
        return H_inter, b_inter

    @staticmethod
    def jacobian_stuff_backward(D_Hiter, D_biter, J_pie, J_Trans, residual, w_final):
        Jacobian = torch.matmul(J_pie, J_Trans)
        Jacobian_w = Jacobian*w_final.unsqueeze(1)

        D_Jacobian_T = torch.matmul(D_Hiter, Jacobian_w.T)
        D_Jacobian_w = torch.matmul(D_Hiter.T, Jacobian.T).T
        D_Jacobian = D_Jacobian_T.T

        D_Jacobian_w_T = torch.matmul(D_biter.unsqueeze(1), residual.unsqueeze(0))
        D_Jacobian_w += D_Jacobian_w_T.T
        D_residual = torch.matmul(D_biter.unsqueeze(1).T, Jacobian_w.T).T

        D_w_final = torch.sum(D_Jacobian_w*Jacobian, dim=-1)
        D_Jacobian += D_Jacobian_w*w_final.unsqueeze(1)

        D_J_pie = torch.matmul(D_Jacobian, J_Trans.T)
        D_J_Trans = torch.matmul(D_Jacobian.T, J_pie).T

        return D_J_pie, D_J_Trans, D_residual.squeeze(1), D_w_final

    @staticmethod
    def perform_single_iter(T_i, T_j, X_j, aij, wj, r_, pix_T_camXs_b):
        w_final = wj*aij
        TjXj = DenseSE3Func.applyRT(T_j, X_j)
        TiXj = DenseSE3Func.applyRT(T_i, X_j)
        pie_TjXj = DenseSE3Func.pie(pix_T_camXs_b, TjXj)
        pie_TiXj = DenseSE3Func.pie(pix_T_camXs_b, TiXj)
        residual = pie_TiXj - (pie_TjXj + r_)
        J_pie = DenseSE3Func.pie_jacobian(TiXj, pix_T_camXs_b, pie_TiXj[2])
        J_T = DenseSE3Func.transformation_jacobian(X_j)

        H_inter, b_inter = DenseSE3Func.jacobian_stuff(J_pie, J_T, residual, w_final)
        return H_inter, b_inter

    @staticmethod
    def perform_single_iter_backward(D_Hiter, D_biter, T_i, T_j, X_j, aij, wj, r_, pix_T_camXs_b):
        
        # D_biter -> 3
        # D_Hiter -> 6x6

        w_final = wj*aij
        TjXj = DenseSE3Func.applyRT(T_j, X_j)
        TiXj = DenseSE3Func.applyRT(T_i, X_j)
        pie_TjXj = DenseSE3Func.pie(pix_T_camXs_b, TjXj)
        pie_TiXj = DenseSE3Func.pie(pix_T_camXs_b, TiXj)
        residual = pie_TiXj - (pie_TjXj + r_)
        J_pie = DenseSE3Func.pie_jacobian(TiXj, pix_T_camXs_b, pie_TiXj[2])
        J_T = DenseSE3Func.transformation_jacobian(X_j)

        D_J_pie, D_J_T, D_residual, D_w_final = DenseSE3Func.jacobian_stuff_backward(D_Hiter, D_biter, J_pie, J_T, residual, w_final)

        D_X_j = DenseSE3Func.transformation_jacobian_backward(D_J_T)
        D_TiXj, D_K, D_dash = DenseSE3Func.pie_jacobian_backward(D_J_pie, TiXj, pix_T_camXs_b, pie_TiXj[2])

        D_pie_TiXj = D_residual.clone()
        D_pie_TiXj[2] += D_dash

        D_pie_TjXj = -D_residual.clone()
        D_r_ = -D_residual.clone()
        # D_r_ = D_r_.squeeze(1)

        D_K1, D_TjXj = DenseSE3Func.pie_backward(D_pie_TjXj, pix_T_camXs_b, TjXj)
        D_K2, D_TiXj2 = DenseSE3Func.pie_backward(D_pie_TiXj, pix_T_camXs_b, TiXj)
        D_TiXj += D_TiXj2 #.squeeze(1)
        D_TjXj = D_TjXj #.squeeze(1)

        DK = D_K + D_K1 + D_K2 
        D_Tj, D_Xj1 = DenseSE3Func.applyRT_backward(D_TjXj, T_j, X_j)
        D_Ti, D_Xj2 = DenseSE3Func.applyRT_backward(D_TiXj, T_i, X_j)

        D_Xj = D_X_j + D_Xj1 + D_Xj2

        D_aij = torch.sum(wj*D_w_final)
        D_wj = D_w_final*aij

        return D_Ti, D_Tj, D_Xj, D_aij, D_wj, D_r_, DK

    @staticmethod
    def least_sq(H_out, b_out):
        return torch.matmul(torch.inverse(H_out), b_out)

    @staticmethod
    def least_sq_backward(D_out, H_out, b_out):
        H_inv = torch.inverse(H_out)
        D_bout = torch.matmul(D_out.unsqueeze(1).T, H_inv).T.squeeze(1)
        D_Hinv = torch.matmul(D_out.unsqueeze(1), b_out.unsqueeze(1).T)
        D_Hout = -1*torch.matmul(H_inv.T, torch.matmul(D_Hinv, H_inv.T))
        return D_Hout, D_bout

    @staticmethod
    def forward(ctx, revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat):
        print("Forward call")
        B, _, H, W = revisions.shape
        delta_twists = torch.zeros((B, H, W, 6)).to(revisions.device).double()
        H_out_all = torch.zeros((B, H, W, 6, 6)).to(revisions.device).double()
        b_out_all = torch.zeros((B, H, W, 6)).to(revisions.device).double()

        for b in range(revisions.shape[0]):
            pix_T_camXs_b = pix_T_camXs[b]
            for i in range(H):
                for j in range(W):
                    H_out = torch.zeros((6, 6)).to(revisions.device).double()
                    b_out = torch.zeros((6)).to(revisions.device).double()
                    for k in range(H):
                        for l in range(W):
                            T_i = Tmat[b, i, j].clone()
                            T_j = Tmat[b, k, l].clone()
                            X_j = xyz_camX[b, k, l].clone()
                            aij = affinity[b:b+1, i, j, k, l].clone()
                            wj = weights[b, :, k, l].clone()
                            r_ = revisions[b, :, k, l].clone()
                            H_iter, b_iter = DenseSE3Func.perform_single_iter(T_i, T_j, X_j, aij, wj, r_, pix_T_camXs_b)
                            
                            H_out = H_out + H_iter
                            b_out = b_out + b_iter
                    
                    H_out_all[b, i, j] = H_out
                    b_out_all[b, i, j] = b_out
                    delta_twists[b,i,j] = DenseSE3Func.least_sq(H_out, b_out)

        ctx.save_for_backward(revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat, H_out_all, b_out_all)
        return delta_twists


    @staticmethod
    def backward(ctx, grad_output):
        # st()
        revisions, weights, xyz_camX, pix_T_camXs, affinity, Tmat, H_out_all, b_out_all = ctx.saved_tensors
        B, _, H, W = revisions.shape
        D_Tmat = torch.zeros_like(Tmat).to(Tmat.device).double()
        D_xyz_camX = torch.zeros_like(xyz_camX).to(xyz_camX.device).double()
        D_affinity = torch.zeros_like(affinity).to(affinity.device).double()
        D_weights = torch.zeros_like(weights).to(weights.device).double()
        D_revisions = torch.zeros_like(revisions).to(revisions.device).double()
        D_pix_T_camXs = torch.zeros_like(pix_T_camXs).to(pix_T_camXs.device).double()

        B = revisions.shape[0]
        for b in range(revisions.shape[0]):
            pix_T_camXs_b = pix_T_camXs[b]
            for i in range(H):
                for j in range(W):
                    D_bij = grad_output[b,i,j]
                    # st()
                    D_Hout, D_bout = DenseSE3Func.least_sq_backward(D_bij, H_out_all[b,i,j], b_out_all[b,i,j])
                    for k in range(H):
                        for l in range(W):
                            T_i = Tmat[b, i, j].clone()
                            T_j = Tmat[b, k, l].clone()
                            X_j = xyz_camX[b, k, l].clone()
                            aij = affinity[b:b+1, i, j, k, l].clone()
                            wj = weights[b, :, k, l].clone()
                            r_ = revisions[b, :, k, l].clone()
                            D_Ti, D_Tj, D_Xj, D_aij, D_wj, D_r_, D_K = DenseSE3Func.perform_single_iter_backward(D_Hout, D_bout, T_i, T_j, X_j, aij, wj, r_, pix_T_camXs_b)
                            D_Tmat[b,i,j] += D_Ti
                            D_Tmat[b,k,l] += D_Tj
                            D_xyz_camX[b,k,l] += D_Xj
                            D_affinity[b,i,j,k,l] += D_aij
                            D_weights[b,:,k,l] += D_wj
                            D_revisions[b,:,k,l] += D_r_
                            D_pix_T_camXs[b] += D_K 
        

        D_Tmat = D_Tmat/B
        D_xyz_camX = D_xyz_camX/B
        D_affinity = D_affinity/B
        D_weights = D_weights/B
        D_revisions = D_revisions/B
        D_pix_T_camXs = D_pix_T_camXs/B
        D_Tmat = D_Tmat/B
        return D_revisions, D_weights, D_xyz_camX, D_pix_T_camXs, D_affinity, D_Tmat




if __name__ == '__main__':
    execute = False
    

    if execute:
        B, H, W = 2, 40, 90
        pix_T_camXs = torch.tensor([[[98.4375,  0.0000, 44.9531,  0.0000],
            [ 0.0000, 77.7778, 19.9630,  0.0000],
            [ 0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.0000]]]).cuda().repeat(B,1,1)
        embeddings = torch.zeros((B, 32, H, W)).cuda()
        revisions = torch.zeros((B, 3, H, W)).cuda()
        weights = torch.zeros_like(revisions).cuda()
        depth = torch.zeros((B, 1, H, W)).cuda()
        
        Tmat = torch.zeros((B, H,W,3,4)).cuda()
        apply_denseSE3_func(embeddings, revisions, weights, depth, pix_T_camXs, Tmat)
    else:
        checkgrad()
    
