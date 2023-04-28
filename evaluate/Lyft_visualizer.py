from typing import List, Tuple
from l5kit.data.filter import (filter_agents_by_frames, filter_agents_by_labels, filter_tl_faces_by_frames)
from l5kit.simulation.unroll import SimulationOutput
from l5kit.visualization.visualizer.common import ( FrameVisualization,  TrajectoryVisualization)
from l5kit.visualization.visualizer.zarr_utils import _get_frame_data,_get_in_out_as_trajectories
import numpy as np



class Visualizer():
    def __init__(self,dataset):
        self.dataset=dataset


    def lyft_visualizer_scene(self, sim_out: SimulationOutput) -> List[FrameVisualization]:
        frames = sim_out.simulated_ego
        agents_frames = filter_agents_by_frames(frames, sim_out.simulated_agents)
        tls_frames = filter_tl_faces_by_frames(frames, sim_out.simulated_dataset.dataset.tl_faces)
     #   agents_th = sim_out.simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]
        ego_ins_outs = sim_out.ego_ins_outs
       # agents_ins_outs = sim_out.agents_ins_outs

        has_ego_info = len(ego_ins_outs) > 0
        #has_agents_info = len(agents_ins_outs) > 0

        frames_vis: List[FrameVisualization] = []

        replay_traj_all=ego_ins_outs[0].inputs['centroid'][None]
        sim_traj_all=ego_ins_outs[0].inputs['centroid'][None]

        scene_index = sim_out.scene_id

        goal_pos = self.dataset.scene_target[scene_index]

        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]
            tls_frame = tls_frames[frame_idx]

            agents_frame = agents_frames[frame_idx]
            # agents_frame = filter_agents_by_labels(agents_frame, agents_th)

            frame_vis = _get_frame_data(self.dataset.meta_manager.map_api, frame, agents_frame, tls_frame)
            trajectories = []

            if has_ego_info:
                ego_in_out = ego_ins_outs[frame_idx]
                replay_traj, sim_traj = _get_in_out_as_trajectories(ego_in_out)


                trajectories.append(TrajectoryVisualization(xs=replay_traj_all[:, 0], ys=replay_traj_all[:, 1],
                                                            color="orange", legend_label="ego_log", track_id=-1))

                trajectories.append(TrajectoryVisualization(xs=sim_traj_all[:, 0], ys=sim_traj_all[:, 1],
                                                            color="red", legend_label="ego_sim", track_id=-1))

                trajectories.append(TrajectoryVisualization(xs=sim_traj[:len(sim_traj)//2, 0], ys=sim_traj[:len(sim_traj)//2, 1],
                                                            color="cyan", legend_label="ego_plan", track_id=-1))

                # trajectories.append(TrajectoryVisualization(xs=sim_traj[len(sim_traj)//2:, 0], ys=sim_traj[len(sim_traj)//2:, 1],
                #                                             color="green", legend_label="ego_pred", track_id=-1))

                replay_traj_all=np.concatenate([replay_traj_all,replay_traj[:1]],axis=0)

                sim_traj_all=np.concatenate([sim_traj_all,sim_traj[:1]],axis=0)

                #centroid = ego_in_out.inputs['centroid']

                # goal = np.concatenate([centroid[None], goal_pos[None]], axis=0)
                #
                # trajectories.append(TrajectoryVisualization(xs=goal[:, 0], ys=goal[:, 1],
                #                                             color="yellow", legend_label="ego_goal", track_id=-1))

                # lane_feature=self.dataset.make_lane_polylines(tls_frame,centroid, scene_index)
                #
                # lane_feature=np.concatenate(lane_feature,axis=0)
                #
                # for i in range(len(lane_feature)):
                #     waypoint = np.stack([lane_feature[i][:2], lane_feature[i][2:4]], axis=0)
                #
                #     trajectories.append(TrajectoryVisualization(xs=waypoint[:, 0], ys=waypoint[:, 1],
                #                                                 color="cyan", legend_label="lane_vector", track_id=-1))

                frame_vis = FrameVisualization(ego=frame_vis.ego, agents=frame_vis.agents,
                                               lanes=frame_vis.lanes, crosswalks=frame_vis.crosswalks,
                                               trajectories=trajectories)

                frames_vis.append(frame_vis)

        return frames_vis


