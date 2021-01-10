from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.sceneflow import SceneFlow
import pydisco_utils
from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets
import ipdb 
st = ipdb.set_trace
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 50
VAL_FREQ = 5000


# def sequence_loss_raft(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
#     """ Loss function defined over sequence of flow predictions """

#     n_predictions = len(flow_preds)    
#     flow_loss = 0.0

#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#     valid = (valid >= 0.5) & (mag < max_flow)

#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         i_loss = (flow_preds[i] - flow_gt).abs()
#         flow_loss += i_weight * (valid[:, None] * i_loss).mean()

#     epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]

#     metrics = {
#         'epe': epe.mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#         '10px': (epe < 10).float().mean().item(),
#         '30px': (epe < 30).float().mean().item(),
#     }

#     return flow_loss, metrics

def get_gt_scene_flow(flow_gt, depth1, depth2):

    B, _, H, W = flow_gt.shape
    ycoords, xcoords = torch.meshgrid(torch.arange(H), torch.arange(W))
    ycoords, xcoords = ycoords.to(flow_gt.device), xcoords.to(flow_gt.device)
    grid = torch.stack([xcoords, ycoords]).unsqueeze(0).repeat(B, 1, 1, 1)
    grid_flowed = grid + flow_gt
    
    inv_depth1 = 1./(depth1 + 1e-5)
    inv_depth2 = 1./(depth2 + 1e-5)

    inv_depth2_sampled = pydisco_utils.grid_sample(inv_depth2, grid_flowed.permute(0,2,3,1))
    inv_depth_change = inv_depth2_sampled - inv_depth1

    scene_flow_gt = torch.cat([flow_gt, inv_depth_change], dim=1)
    return scene_flow_gt

def sequence_loss(motion_preds, flow_gt, valid, depth1, depth2, pix_T_camXs, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    scene_flow_gt = flow_gt #get_gt_scene_flow(flow_gt, depth1, depth2)
    scene_flow_gt = scene_flow_gt.permute(0,3,1,2)
    n_predictions = len(motion_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # st()
    # valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred_i, coords1_i, d_dash_i = pydisco_utils.get_flow_field(depth1, motion_preds[i].permute(0,2,3,1), pix_T_camXs)
        inv_depth_change = d_dash_i - (1./(depth1 + 1e-5))
        flow_pred_i = flow_pred_i.permute(0,3,1,2)
        scene_flow_pred_i = torch.cat([flow_pred_i, inv_depth_change], dim=1)

        i_loss = (scene_flow_pred_i - scene_flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()


    epe = torch.sum((scene_flow_pred_i[:,:2] - scene_flow_gt[:,:2])**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        '10px': (epe < 10).float().mean().item(),
        '30px': (epe < 30).float().mean().item(),
        'loss': flow_loss.item()
    }
    print("flow loss: ", flow_loss)

    if torch.isnan(flow_loss):
        st()
        aa=1
    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push_gradients(self, named_params):
        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            if self.writer is None:
                self.writer = SummaryWriter()
            # st()
            for n, p in named_params:
                if p.requires_grad and 'bias' not in n and p.grad!=None and not torch.isnan(p.grad.mean()):
                    print("name, gradmean, steps: ", n, p.grad.abs().mean(), self.total_steps)
                    self.writer.add_histogram("grads/"+n+"_hist", p.grad.reshape(-1), self.total_steps) 
                    self.writer.add_scalar("grads/"+n+"_scal", p.grad.abs().mean(), self.total_steps) 

    def push_dict(self, named_params):
        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            if self.writer is None:
                self.writer = SummaryWriter()
            # st()
            for n, p in named_params.items():
                print("name, outmean, steps: ", n, p.abs().mean(), self.total_steps)
                self.writer.add_histogram("output/"+n+"_hist", p.reshape(-1), self.total_steps) 
                self.writer.add_scalar("output/"+n+"_scal", p.abs().mean(), self.total_steps) 


    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def normalize_image(image):
    image = image[:, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def fetch_dataloader(args):
    gpuargs = {'shuffle': True, 'num_workers': 4, 'drop_last' : True}
    train_dataset = SceneFlow(do_augment=True, image_size=[320, 768])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **gpuargs)
    return train_loader

def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    # train_loader = datasets.fetch_dataloader(args)
    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # image1, image2, disp1, disp2, flow, valid = [x.cuda() for x in data_blob]
            image1, image2, depth1, depth2, flowxyz, intrinsics = [x.cuda() for x in data_blob]
            pix_T_camXs = pydisco_utils.sceneflow_intrinsics_to_pydisco(intrinsics)
            depth1, depth2 = depth1.unsqueeze(1), depth2.unsqueeze(1)
            # B = image1.shape[0]
            # pix_T_camXs = pydisco_utils.get_pix_T_camX(args.stage, image1.shape[0])
            # depth1 = pydisco_utils.disp2depth(torch.ones(B).cuda(), pix_T_camXs[:,0,0], disp1)
            # depth2 = pydisco_utils.disp2depth(torch.tensor(B).cuda(), pix_T_camXs[:,0,0], disp2)
            
            # image1, image2, depth1, depth2, flow, valid, pix_T_camXs = rescale_stuff(image1, image2, depth1, depth2, flow, valid, pix_T_camXs)
            # valid = (flowxyz[:, 0].abs() < 720) & (flowxyz[:, 1].abs() < 720)
            valid = (depth1 < 255.0)
            # valid = valid*valid_mask.squeeze()


            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            motion_predictions, return_dict = model(image1, image2, depth1, depth2, pix_T_camXs, iters=args.iters)      
            loss, metrics = sequence_loss(motion_predictions, flowxyz, valid, depth1, depth2, pix_T_camXs, args.gamma)
            
            # loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            logger.push_gradients(model.named_parameters())
            logger.push_dict(return_dict)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH

def rescale_stuff(img1, img2, depth1, depth2, flow, valid, pix_T_camX):

    finalH, finalW = 320, 720
    sy = finalH/img1.shape[2]
    sx = finalW/img1.shape[3]
    img1 = F.interpolate(img1, (finalH, finalW), mode="bilinear")
    img2 = F.interpolate(img2, (finalH, finalW), mode="bilinear")
    depth1 = F.interpolate(depth1, (finalH, finalW), mode="nearest")
    depth2 = F.interpolate(depth2, (finalH, finalW), mode="nearest")
    flow = F.interpolate(flow, (finalH, finalW), mode="bilinear")
    flow = flow * torch.tensor([sx, sy]).reshape(1,2,1,1).to(flow.device)

    valid = (flow[:, 0].abs() < 720) & (flow[:, 1].abs() < 720)
    valid_mask = (depth1 < 255.0)
    valid = valid*valid_mask.squeeze()

    pix_T_camX = pydisco_utils.scale_intrinsics(pix_T_camX, sx, sy)
    return img1, img2, depth1, depth2, flow, valid, pix_T_camX

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)