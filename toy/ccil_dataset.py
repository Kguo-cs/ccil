import torch
from torch.utils.data import Dataset

import numpy as np
from l5kit.geometry import angular_distance, compute_agent_pose,rotation33_as_yaw,yaw_as_rotation33
from l5kit.geometry.transform import transform_points,transform_point
from torch.utils.data import DataLoader


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

def transform(history_position,radius,centroid_std=1):
    inputs=[]

    for i in range(len(history_position)-1,-1,-1):

        ego_centroid_m = history_position[i]+ np.random.randn(2) *  centroid_std

        vector = np.zeros([2]) - ego_centroid_m

        agent_yaw_rad = np.arctan2(vector[1], vector[0])

        world_from_agent = compute_agent_pose(ego_centroid_m, agent_yaw_rad)

        agent_from_world = np.linalg.inv(world_from_agent)

        nearest_point_on_circle=ego_centroid_m/np.linalg.norm(ego_centroid_m)*radius

        rel_pos = transform_point(nearest_point_on_circle, agent_from_world)

        inputs.append(rel_pos)

    inputs=np.stack(inputs,axis=0)

    return inputs,world_from_agent,agent_from_world,agent_yaw_rad


class CcilDataset(Dataset):

    def __init__(self,):

        self.speed=1


    def __len__(self):
        return 1000000000

    def __getitem__(self, index) -> dict:

        cur_theta = np.random.random() * 2 * np.pi-np.pi

        radius=np.random.random()*90+10

        history_position, history_yaw,angular_speed=get_history(radius,cur_theta)

        inputs,world_from_agent,agent_from_world,agent_yaw_rad=transform(history_position,radius)

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
    dataset=CcilDataset()

    dataloader = DataLoader(dataset,batch_size=8)
    return dataloader