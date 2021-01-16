import torch
import numpy as np
import torch.nn.functional as F

def get_filtered_trajs(trajs_XYs, trajs_Ts, min_lifespan, min_dist=0, trajs_XYZs=None):
    trajs_XYs_filtered = []
    trajs_Ts_filtered = []
    if trajs_XYZs is not None:
        trajs_XYZs_filtered = []
    for i in range(len(trajs_XYs)):
        traj = trajs_XYs[i]
        endpoint_dist = torch.norm(traj[-1] - traj[0])
        if len(trajs_XYs[i]) >= min_lifespan:
            if min_dist==0:
                trajs_XYs_filtered.append(trajs_XYs[i])
                trajs_Ts_filtered.append(trajs_Ts[i])
                if trajs_XYZs is not None:
                    trajs_XYZs_filtered.append(trajs_XYZs[i])
            elif endpoint_dist >= min_dist:
                trajs_XYs_filtered.append(trajs_XYs[i])
                trajs_Ts_filtered.append(trajs_Ts[i])
                if trajs_XYZs is not None:
                    trajs_XYZs_filtered.append(trajs_XYZs[i])
    if trajs_XYZs is not None:
        return trajs_XYs_filtered, trajs_XYZs_filtered, trajs_Ts_filtered
    else:
        return trajs_XYs_filtered, trajs_Ts_filtered