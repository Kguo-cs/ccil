import torch.nn as nn
import torch
from .lqr_solver import LQR_solver

class LQR(nn.Module):
    def __init__(self, cfg):
        super(LQR, self).__init__()

        future_num_frames = cfg['future_num_frames']

        acc_w = cfg["acc_w"]

        control_w = cfg["control_w"]

        step_time=cfg['step_time']

        yaw_vel_w = cfg["yaw_vel_w"]

        yaw_acc_w = cfg["yaw_acc_w"]

        yaw_control_w=cfg["yaw_control_w"]

        self.yaw_w=cfg["yaw_w"]

        self.n_state = 9

        self.n_ctrl = 3

        C = torch.zeros([future_num_frames + 1, 12, 12])

        C[1:, 0, 0] = 1
        C[1:, 1, 1] = 1
        C[1:, 2, 2] = self.yaw_w

        C[1:, 5, 5] = yaw_vel_w

        C[1:, 6, 6] = acc_w
        C[1:, 7, 7] = acc_w
        C[1:, 8, 8] = yaw_acc_w

        C[1:, 9, 9] = control_w
        C[1:, 10, 10] = control_w
        C[1:, 11, 11] = yaw_control_w

        F = torch.zeros([9, 12])

        for i in range(9):
            F[i][i] = 1
            F[i][i + 3] = step_time

        for i in range(6):
            F[i][i + 6] = step_time * step_time

        for i in range(3):
            F[i][i + 9] = step_time * step_time * step_time

        self.F=F[None][None]

        self.C=C[:,None]

        self.LQR_solver=LQR_solver(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=future_num_frames + 1,
            u_lower=-1000,
            u_upper=1000,
            )

    def forward(self,ego_polyline,action_preds):
        last_action_preds = action_preds[:, -1]

        x_init = ego_polyline[..., :self.n_state]

        target= last_action_preds[..., :self.n_ctrl]

        plan = self.solve(target, x_init)

        return plan

    def solve(self,target,x_init):
        n_batch,t,n_ctrl=target.shape

        T=t+1

        C=self.C.repeat(1,n_batch,1,1).to(x_init.device)

        c = torch.zeros([T, n_batch, self.n_state + self.n_ctrl]).to(x_init.device)  # target_state

        c[1:,:,:n_ctrl]=-target.permute(1,0,2)

        c[1:,:,n_ctrl-1]*=self.yaw_w

        F=self.F.repeat(T,n_batch,1,1).to(x_init.device)

        x_lqr=self.LQR_solver.solve(x_init,C,c,F)

        return x_lqr[1:,:,:n_ctrl].permute(1,0,2)


