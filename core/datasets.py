# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from rutils import frame_utils
from rutils.augmentor import FlowAugmentor, SparseFlowAugmentor
import ipdb 
st = ipdb.set_trace

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        # self.augmentor = None
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def rescale_stuff(self, img1_np, img2_np, flow_np):
        sy = 400.0/540
        sx = 720.0/960

        img1 = torch.tensor(img1_np).unsqueeze(0).permute(0,3,1,2).float()
        img2 = torch.tensor(img2_np).unsqueeze(0).permute(0,3,1,2).float()
        flow = torch.tensor(flow_np).unsqueeze(0).permute(0,3,1,2).float()

        img1 = F.interpolate(img1, size=(400, 720), mode="bilinear")
        img2 = F.interpolate(img2, size=(400, 720), mode="bilinear")
        flow = F.interpolate(flow, (400, 720), mode="bilinear")
        flow = flow * torch.tensor([sx, sy]).reshape(1,2,1,1)

        return img1.permute(0,2,3,1)[0].numpy(), img2.permute(0,2,3,1)[0].numpy(), flow.permute(0,2,3,1)[0].numpy()

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        else:
            img1, img2, flow = self.rescale_stuff(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class FlowTrajDataset(data.Dataset):
    def __init__(self, args, aug_params=None, sparse=False):
        self.augmentor = None
        self.args = args
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.trajlen = args.trajlen
        self.augmentor = None
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.imglist_traj = []
        self.flowfw_traj = []
        self.flowbk_traj = []

    def rescale_stuff(self, img1_np, flow_fw_np, flow_bk_np):
        sy = 400.0/540
        sx = 720.0/960

        img1 = torch.tensor(img1_np).unsqueeze(0).permute(0,3,1,2).float()
        flow_fw = torch.tensor(flow_fw_np).unsqueeze(0).permute(0,3,1,2).float()
        flow_bk = torch.tensor(flow_bk_np).unsqueeze(0).permute(0,3,1,2).float()

        img1 = F.interpolate(img1, size=(400, 720), mode="bilinear")

        flow_fw = F.interpolate(flow_fw, (400, 720), mode="bilinear")
        flow_fw = flow_fw * torch.tensor([sx, sy]).reshape(1,2,1,1)

        flow_bk = F.interpolate(flow_bk, (400, 720), mode="bilinear")
        flow_bk = flow_bk * torch.tensor([sx, sy]).reshape(1,2,1,1)

        return img1.permute(0,2,3,1)[0].numpy(), flow_fw.permute(0,2,3,1)[0].numpy(), flow_bk.permute(0,2,3,1)[0].numpy()

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.imglist_traj)
        valid = None
        imagelist_traj_idx = self.imglist_traj[index]
        flow_fw_idx = self.flowfw_traj[index]
        flow_bk_idx = self.flowbk_traj[index]

        # total_imgs_in_traj = len(imagelist_traj_idx)

        # For now lets keep it simple and take first traj_len samples. # TODO: randomly sample starting frame.
        img_list = []
        flow_fw_list = []
        flow_bk_list = []
        valid_fw_list = []
        valid_bk_list = []
        for i in range(self.trajlen):
            
            if self.sparse:
                st() # this condition is not required. Fix this before using.
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                if i < self.trajlen-1:
                    flow_fw = frame_utils.read_gen(flow_fw_idx[i])
                    flow_bk = frame_utils.read_gen(flow_bk_idx[i])
                else:
                    # These are just placeholders for processing and won't be used.
                    flow_fw = frame_utils.read_gen(flow_fw_idx[i-1])
                    flow_bk = frame_utils.read_gen(flow_bk_idx[i-1])

            img1 = frame_utils.read_gen(imagelist_traj_idx[i])
            
            flow_fw = np.array(flow_fw).astype(np.float32)
            flow_bk = np.array(flow_bk).astype(np.float32)

            img1 = np.array(img1).astype(np.uint8)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[...,None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
            
            # TODO: decide how to modify augmentors for the trajectory. Do we need occlusions?
            if self.augmentor is not None:
                if self.sparse:
                    st() # Fix this before using.
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)
            else:
                img1, flow_fw, flow_bk = self.rescale_stuff(img1, flow_fw, flow_bk)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            flow_fw = torch.from_numpy(flow_fw).permute(2, 0, 1).float()
            flow_bk = torch.from_numpy(flow_bk).permute(2, 0, 1).float()

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid_fw = (flow_fw[0].abs() < 1000) & (flow_fw[1].abs() < 1000)
                valid_bk =  (flow_bk[0].abs() < 1000) & (flow_bk[1].abs() < 1000)

            img_list.append(img1)
            if i < self.trajlen-1: 
                flow_fw_list.append(flow_fw)
                flow_bk_list.append(flow_bk)
                valid_fw_list.append(valid_fw)
                valid_bk_list.append(valid_bk)

        
        img1 = torch.stack(img_list)
        flow_fw = torch.stack(flow_fw_list)
        flow_bk = torch.stack(flow_bk_list)
        valid_fw = torch.stack(valid_fw_list)
        valid_bk = torch.stack(valid_bk_list)

        return img1, flow_fw, flow_bk, valid_fw.float(), valid_bk.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.imglist_traj)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
        
        print("Length of flow list: ", len(self.flow_list))
        print("Length of img list: ", len(self.image_list))

class FlyingThingsTraj3D(FlowTrajDataset):
    def __init__(self, args, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThingsTraj3D, self).__init__(args, aug_params)

        for cam in ['left']:
            image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs_root = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
            flow_dirs_fw = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs_root])
            flow_dirs_bk = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs_root])

            for idir, fdir, fdir_bk in zip(image_dirs, flow_dirs_fw, flow_dirs_bk):
                images = sorted(glob(osp.join(idir, '*.png')) )
                flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                flows_bk = sorted(glob(osp.join(fdir_bk, '*.pfm')) )

                self.imglist_traj.append(images)
                self.flowfw_traj.append(flows[:-1])
                self.flowbk_traj.append(flows_bk[1:])
                

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == "thingstraj":
        print("Loading flying things 3d trajectory dataset")
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThingsTraj3D(args, aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThingsTraj3D(args, aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

