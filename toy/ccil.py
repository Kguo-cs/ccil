import numpy as np

from .ccil_dataset import get_dataloader,get_history,transform
from l5kit.geometry import  transform_points

from torch import nn
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
from torch.nn import functional as F

class MLP(nn.Module):
    r"""
    Construct a MLP, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, hidden_size,output_size):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, ..., input_size]
        """
        x=x.reshape(-1,20)

        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# class MLP(nn.Module):
#     r"""
#     Construct a MLP, include a single fully-connected layer,
#     followed by layer normalization and then ReLU.
#     """
#
#     def __init__(self, input_size, hidden_size,output_size):
#         r"""
#         self.norm is layer normalization.
#         Args:
#             input_size: the size of input layer.
#             output_size: the size of output layer.
#             hidden_size: the size of output layer.
#         """
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.norm = torch.nn.LayerNorm(hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.causal_len=10
#
#         self.embed_timestep = nn.Embedding(self.causal_len,hidden_size)
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             dim_feedforward=hidden_size,
#             nhead=8,
#             dropout=0.1,
#             batch_first=True)
#
#
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#
#
#     def forward(self, x):
#         r"""
#         Args:
#             x: x.shape = [batch_size, ..., input_size]
#         """
#         x = self.fc1(x)
#         x = self.norm(x)
#         timestep=torch.arange(self.causal_len,device=device)
#         time_embed=self.embed_timestep(timestep)
#
#         x+=time_embed[None]
#
#         x=self.transformer(x)
#         x = self.fc2(x[:,-1])
#         return x


def closed_loop_eval():

    batch = []

    world_from_agents=[]

    agent_yaw_rads=[]

    for i in range(traj_num):

        cur_theta = np.random.random() * 2 * np.pi - np.pi

        history_position, history_yaw,angular_speed = get_history(radius, cur_theta)

        inputs,world_from_agent,agent_from_world,agent_yaw_rad=transform(history_position,radius,centroid_std=0)

        batch.append(inputs)

        world_from_agents.append(world_from_agent)

        agent_yaw_rads.append(agent_yaw_rad)

    batch=np.stack(batch,axis=0)

    trajectories=np.zeros([traj_len,traj_num,3])

    world_from_agents=np.stack(world_from_agents,axis=0)

    agent_yaw_rads=np.stack(agent_yaw_rads,axis=0)

    for iteration in range(traj_len):

        history=batch[:,1:]

        batch=torch.tensor(batch).to(torch.float32).to(device)

        outputs=model(batch).cpu().numpy()

        new_yaw=agent_yaw_rads+outputs[:,-1:]

        new_pos=transform_points(outputs[:,None,:2], world_from_agents)[:,0]

        new_state=np.concatenate([new_pos,new_yaw],axis=-1)

        trajectories[iteration]=new_state

        batch=[]

        world_from_agents=[]

        agent_yaw_rads=[]

        for i in range(traj_num):

            inputs,world_from_agent,agent_from_world,agent_yaw_rad=transform(new_pos[i,None],radius,centroid_std=0)

            batch.append(inputs)

            world_from_agents.append(world_from_agent)

            agent_yaw_rads.append(agent_yaw_rad)

        batch=np.stack(batch,axis=0)

        batch=np.concatenate([history,batch],axis=1)

        world_from_agents = np.stack(world_from_agents, axis=0)

        agent_yaw_rads = np.stack(agent_yaw_rads, axis=0)

    return trajectories



device=torch.device("cuda:0")

data_loader=get_dataloader()

radius = 50

traj_num = 1

traj_len = 100

initial_num=100

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

circle = plt.Circle((0, 0), radius, edgecolor='black', fill=False)
ax.add_patch(circle)


all_trajectories=np.zeros([initial_num,traj_len,traj_num,3])

l1loss = nn.L1Loss(reduction="mean")

for n in tqdm(range(initial_num)):

    model=MLP(input_size=20,hidden_size=64,output_size=3).to(device)

    optim=torch.optim.AdamW( model.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )

    for step,batch in enumerate(data_loader):

        inputs=batch["inputs"].to(torch.float32).to(device)
        targets=batch["targets"].to(torch.float32).to(device)

        outputs=model(inputs)

        optim.zero_grad()

        loss=l1loss(targets,outputs)

        loss.backward()

        optim.step()

        #print(step,loss.item())
#
        if step==2000:
            #print(loss.item())

            with torch.no_grad():
                model.eval()

                trajectories=closed_loop_eval()

                for num in range(traj_num):
                    plt.plot(trajectories[:, num, 0], trajectories[:, num, 1])
                all_trajectories[n]=trajectories

                model.train()
            break

plt.xlim([-radius*2, radius*2])
plt.ylim([-radius*2, radius*2])
np.save("ccil.npy",all_trajectories)
plt.show()









