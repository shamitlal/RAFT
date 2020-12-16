import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder, BasicEncoderRaft3D
from denseSE3 import DenseSE3
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
import pydisco_utils
import ipdb 
st = ipdb.set_trace

import se3_utils
from liegroups.torch import SE3, SO3

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128*3
            args.corr_levels = 4
            args.corr_radius = 4

        self.denseSE3 = DenseSE3()
        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.cnet = BasicEncoderRaft3D() 
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def initialize_transformation(self, img):

        B, _, H, W = img.shape
        zero_rot = pydisco_utils.eye_3x3(1)
        zero_trans = torch.zeros((1, 3)).to(img.device)
        rt = pydisco_utils.merge_rt(zero_rot, zero_trans)
        rt = rt.unsqueeze(1).unsqueeze(1).repeat(B, H//8, W//8, 1, 1)
        return rt
        # return torch.zeros((B, H//8, W//8, 3)).to(img.device)

    def upsample_twist(self, flow, mask):
        """ Upsample twist field [H/8, W/8, 6] -> [H, W, 6] using convex combination """
        N, _, H, W = flow.shape
        # st()
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 6, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 6, 8*H, 8*W)


    def forward(self, image1, image2, depth1_fullres, depth2_fullres, pix_T_camXs_fullres, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        depth1 = F.interpolate(depth1_fullres, scale_factor=1./8, mode="nearest")
        depth2 = F.interpolate(depth2_fullres, scale_factor=1./8, mode="nearest")

        inv_depth1 = 1./(depth1 + 1e-5)
        inv_depth2 = 1./(depth2 + 1e-5)

        pix_T_camXs = pydisco_utils.scale_intrinsics(pix_T_camXs_fullres, 1./8, 1./8)

        hdim = self.hidden_dim  
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)


        Tmat = self.initialize_transformation(image1) # This is BHW44

        motion_predictions = []

        for itr in range(iters):
            flow, coords1, d_dash = pydisco_utils.get_flow_field(depth1, Tmat, pix_T_camXs)
            d_dash_bar = pydisco_utils.grid_sample(inv_depth2, coords1)
            redidual_depth = d_dash - d_dash_bar 
            twist = se3_utils.logmap(Tmat)

            # coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            with autocast(enabled=self.args.mixed_precision):
                net, revisions, weights, embeddings, up_mask = self.update_block(net, inp, corr, flow, redidual_depth, twist)
            
            Tmat = self.denseSE3(embeddings, revisions, weights, depth1, pix_T_camXs, Tmat)
            twist = se3_utils.logmap(Tmat)

            twist = twist.permute(0, 3, 1, 2)
            twists_up = self.upsample_twist(twist, up_mask)
            transformation_up = se3_utils.expmap(twists_up)
            motion_predictions.append(transformation_up)

        if test_mode:
            return motion_predictions, twist
            
        return motion_predictions
