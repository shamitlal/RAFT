from torch.utils.tensorboard import SummaryWriter
import torch 
import numpy as np 
import flow_vis
import ipdb 
st = ipdb.set_trace
import cv2 

class Logger:
    def __init__(self, model, scheduler, sum_freq):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.sum_freq = sum_freq

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push_gradients(self, named_params):
        if self.total_steps % self.sum_freq == self.sum_freq-1:
            if self.writer is None:
                self.writer = SummaryWriter()
            # st()
            for n, p in named_params:
                if p.requires_grad and 'bias' not in n and p.grad!=None and not torch.isnan(p.grad.mean()):
                    print("name, gradmean, steps: ", n, p.grad.abs().mean(), self.total_steps)
                    self.writer.add_histogram("grads/"+n+"_hist", p.grad.reshape(-1), self.total_steps) 
                    self.writer.add_scalar("grads/"+n+"_scal", p.grad.abs().mean(), self.total_steps) 

    def push_trajs_single(self, images, flow_fw, X, Y):
        trajidx = np.random.randint(X.shape[1])
        img_list = []
        for s in range(images.shape[0]):
            imgtouse = np.array(images[s].permute(1,2,0).detach().cpu().numpy()).astype(np.float64)
            aa = imgtouse.copy()
            img_cir = cv2.circle(aa, (X[s][trajidx], Y[s][trajidx]), radius=10, color=(255,0,0), thickness=-1)
            img_list.append(torch.tensor(img_cir).permute(2,0,1))
        img = torch.cat(img_list, dim=-1)
        img = img.unsqueeze(0)
        self.push_rgb("traj", img)


    def push_trajs(self, images, flow_fw, trajsXY):
        if self.total_steps % self.sum_freq == self.sum_freq-1:
            self.push_trajs_single(images[0], flow_fw[0], trajsXY[0][0], trajsXY[0][1])

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

    # flow -> BHWC
    def push_flow(self, name, flow):
        # flow = torch.clamp(flow, 0)
        # try:
        flow_rgb_vis = flow_vis.flow_to_color(flow[0].cpu().detach().numpy(), convert_to_bgr=False)*1.0
        flow_rgb_vis = torch.tensor(flow_rgb_vis).unsqueeze(0)
        self.push_rgb(name, flow_rgb_vis.permute(0,3,1,2))
        # except Exception as e:
        #     print("Exception in flow visualization")

    def push_rgb(self, name, rgb):

        if self.total_steps % self.sum_freq == self.sum_freq-1:
            if self.writer is None:
                self.writer = SummaryWriter()
            self.writer.add_image(name, rgb[0]/255., self.total_steps)

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()