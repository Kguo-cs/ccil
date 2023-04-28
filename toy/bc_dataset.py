import torch
from torch.utils.data import Dataset

import numpy as np
from l5kit.geometry import angular_distance, compute_agent_pose,rotation33_as_yaw,yaw_as_rotation33
from l5kit.geometry.transform import transform_points,transform_point
from torch.utils.data import DataLoader
import torch
import pathlib
import os


def get_history(radius,cur_theta,speed=1):
    angular_speed=speed/radius

    history_position = []

    history_yaw = []

    for i in range(10):
        history_theta = cur_theta - angular_speed * i

        x = np.cos(history_theta) * radius

        y = np.sin(history_theta) * radius

        history_position.append([x, y])

        history_yaw.append(history_theta + np.pi / 2)

    history_position = np.array(history_position)

    history_yaw = np.array(history_yaw)

    return history_position, history_yaw,angular_speed

def transform(history_position,history_yaw,radius):

    ego_centroid_m = history_position[0]

    agent_yaw_rad = history_yaw[0]

    world_from_agent = compute_agent_pose(ego_centroid_m, agent_yaw_rad)

    agent_from_world = np.linalg.inv(world_from_agent)

    rel_pos = transform_points(history_position, agent_from_world)

    rel_yaw = angular_distance(history_yaw[:, None], agent_yaw_rad)

    inputs = np.concatenate([rel_yaw, rel_pos], axis=-1).astype(np.float32)

    nearest_point_on_circle = ego_centroid_m / np.linalg.norm(ego_centroid_m) * radius

    rel_nearest_point = transform_point(nearest_point_on_circle, agent_from_world)

    inputs = np.concatenate([inputs.reshape(-1), rel_nearest_point], axis=-1)

    return inputs,world_from_agent,agent_from_world,agent_yaw_rad


class BcDataset(Dataset):

    def __init__(self,):

        self.speed=1


    def __len__(self):
        return 10000000

    def __getitem__(self, index) -> dict:

        cur_theta = np.random.random() * 2 * np.pi-np.pi

        radius=np.random.random()*90+10

        history_position, history_yaw,angular_speed=get_history(radius,cur_theta)

        inputs,world_from_agent,agent_from_world,agent_yaw_rad=transform(history_position,history_yaw,radius)

        fut_theta=cur_theta + angular_speed

        x = np.cos(fut_theta) * radius

        y = np.sin(fut_theta ) * radius

        fut_position=np.array([x,y])

        fut_yaw=np.array([fut_theta+np.pi/2])

        target_pos=transform_point(fut_position, agent_from_world)

        target_yaw = fut_yaw- agent_yaw_rad

        targets=np.concatenate([target_pos,target_yaw],axis=-1).astype(np.float32)


        data = {
            "inputs": inputs,
            "targets": targets
        }

        return data

def get_dataloader():
    dataset=BcDataset()

    dataloader = DataLoader(dataset,batch_size=8)
    return dataloader