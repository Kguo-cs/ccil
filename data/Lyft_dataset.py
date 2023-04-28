import numpy as np
import bisect

from torch.utils.data import Dataset

from typing import Dict,  Optional

from l5kit.geometry.transform import transform_points
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

from l5kit.data import  filter_agents_by_distance, filter_tl_faces_by_status,get_frames_slice_from_scenes
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw,transform_point,angular_distance
from l5kit.sampling.agent_sampling import get_agent_context, get_relative_poses


class LyftDataset(Dataset):
    def __init__(self, cfg, type, meta_manager, zarr_dataset, scene_dataset=False):

        self.dataset = zarr_dataset

        self.cfg = cfg

        model_cfg=self.cfg['model_params']

        data_cfg=self.cfg['data_generation_params']

        map_cfg=data_cfg["lane_params"]

        self.step_time =model_cfg ['step_time']

        self.causal_len = model_cfg['causal_len']

        self.causal_interval = model_cfg['causal_interval']

        self.history_num_frames_agents = model_cfg["history_num_frames_agents"]
        self.history_num_frames_ego=model_cfg["history_num_frames_ego"]
        self.history_num_frames_max = max(self.history_num_frames_ego, self.history_num_frames_agents)
        self.future_num_frames = model_cfg["future_num_frames"]

        self.ignore_ego=model_cfg["ignore_ego"]

        self.lane_feat_dim=model_cfg["lane_feat_dim"]+1
        self.cross_feat_dim=model_cfg["cross_feat_dim"]+1

        if self.ignore_ego:
            self.ego_feat_dim=  3
        else:
            self.ego_feat_dim=  12

        self.agent_feat_dim =model_cfg["agent_feat_dim"]
        self.agent_all_feat_dim=self.agent_feat_dim*(self.history_num_frames_agents+1)


        self.dist_enhanced=data_cfg["dist_enhanced"]
        self.start_frame_index = data_cfg['start_frame_index']

        self.centroid_std=data_cfg["centroid_std"]
        self.use_goal_yaw=data_cfg["use_goal_yaw"]
        self.use_uniform_yaw=data_cfg["use_uniform_yaw"]
        self.use_uniform_centroid=data_cfg["use_uniform_centroid"]
        self.use_AckermanPerturbation=data_cfg["use_AckermanPerturbation"]

        self.other_agents_num = data_cfg["other_agents_num"]
        self.max_agents_distance = data_cfg["max_agents_distance"]

        self.max_points_per_lane = map_cfg["max_points_per_lane"]
        self.max_points_per_crosswalk= map_cfg["max_points_per_crosswalk"]
        self.max_retrieval_distance = map_cfg["max_retrieval_distance_m"]
        self.max_num_lanes=map_cfg["max_num_lanes"]
        self.max_num_crosswalks=map_cfg["max_num_crosswalks"]

        if scene_dataset == False:

            if self.use_AckermanPerturbation:
                mean = np.array([0.0, 0.0, 0.0])

                std = np.array([0.5, 1.5, np.pi / 6])

                self.perturbation = AckermanPerturbation(
                    random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=0.5)

            self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

            cumulative_sizes = np.concatenate([np.array([0]), self.cumulative_sizes])

            self.scene_target,self.target_index=meta_manager.get_scenetargets(type, self.dataset.frames,  cumulative_sizes)

            self.meta_manager = meta_manager

            self.avail_index = []

            list_cumulative_sizes = [0] + self.cumulative_sizes.tolist()

            start_state_index = self.start_frame_index

            for i in range(len(list_cumulative_sizes) - 1):
                index = np.arange(list_cumulative_sizes[i] + start_state_index, list_cumulative_sizes[i + 1]-1)

                self.avail_index.append(index)

            self.avail_index = np.concatenate(self.avail_index)

    def __len__(self) -> int:
         return len(self.avail_index)

    def __getitem__(self, index) -> dict:

        index=self.avail_index[index]

        lane_polylines=[]
        crosswalk_polylines=[]
        agent_polylines=[]
        ego_polyline=[]

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]

        target_positions = np.zeros([self.causal_len, self.future_num_frames,2], np.float32)
        target_yaws = np.zeros([self.causal_len, self.future_num_frames,1], np.float32)
        target_availabilities = np.zeros([self.causal_len, self.future_num_frames], np.bool)

        for t in range(self.causal_len):
            frame_index = state_index + (-self.causal_len + 1 + t) * self.causal_interval

            if frame_index<0:

                crosswalk_polylines.append( np.ones([self.max_num_crosswalks, self.max_points_per_crosswalk, self.cross_feat_dim], np.float32))
                lane_polylines.append(np.ones([self.max_num_lanes, self.max_points_per_lane, self.lane_feat_dim],    np.float32))
                agent_polylines.append(np.ones([self.other_agents_num, self.agent_all_feat_dim], np.float32))
                ego_polyline.append(np.zeros([1, self.ego_feat_dim], np.float32))
            else:
                lane_polylines_t,crosswalk_polylines_t,agent_polylines_t,ego_polyline_t,future_coords_offset,future_yaws_offset,target_availabilities_t = self.get_frame(scene_index, frame_index, self.dataset)

                lane_polylines.append(lane_polylines_t)
                crosswalk_polylines.append(crosswalk_polylines_t)
                agent_polylines.append(agent_polylines_t)
                ego_polyline.append(ego_polyline_t)

                target_positions[t]=future_coords_offset
                target_yaws[t]=future_yaws_offset
                target_availabilities[t]=target_availabilities_t

        data = {
            "lane_polylines": np.array(lane_polylines),
            "crosswalk_polylines": np.array(crosswalk_polylines),
            "agent_polylines": np.array(agent_polylines),
            "ego_polyline": np.array(ego_polyline),
            "target_availabilities": target_availabilities,
            "target_positions": target_positions,
            "target_yaws": target_yaws
        }

        return data

    def get_frame(self, scene_index: int, frame_index: int, dataset, selected_track_id: Optional[int] = None):

        uniform_noise = 0

        if len(dataset.scenes) == 1:# test
            frames = dataset.frames[get_frames_slice_from_scenes(dataset.scenes[0])]
            centroid_noise =0
        else:
            frames = dataset.frames[get_frames_slice_from_scenes(dataset.scenes[scene_index])]
            centroid_noise=np.random.randn(2)*self.centroid_std
            if self.use_uniform_yaw:
                uniform_noise = np.random.uniform(low=-np.pi / 3, high=np.pi / 3)

        (
            history_frames,
            future_frames,
            history_agents,
            future_agents,
            history_tl_faces,
            future_tl_faces,
        ) = get_agent_context(frame_index, frames, dataset.agents, dataset.tl_faces, self.history_num_frames_max, self.future_num_frames)

        if  len(dataset.scenes) != 1 and len(future_frames) == self.future_num_frames and self.use_AckermanPerturbation:
            history_frames, future_frames = self.perturbation.perturb(
                history_frames=history_frames, future_frames=future_frames
            )

        cur_frame = history_frames[0]

        agent_centroid_m = cur_frame["ego_translation"][:2]+centroid_noise

        goal_translation=self.scene_target[scene_index]

        if self.use_goal_yaw:

            vector=goal_translation-agent_centroid_m

            agent_yaw_rad=np.arctan2(vector[1],vector[0])
        else:

            agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])

            agent_yaw_rad = agent_yaw_rad+uniform_noise

        world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)

        agent_from_world = np.linalg.inv(world_from_agent)

        goal_pos = transform_point(goal_translation, agent_from_world).astype(np.float32)

        future_coords_offset, future_yaws_offset, future_extents, future_availability = get_relative_poses(
            self.future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
        )

        agent_polylines = self.make_agent_polylines(agent_centroid_m,agent_yaw_rad,agent_from_world,
                                                                   history_frames,history_agents)

        lane_polylines = self.make_lane_polylines(history_tl_faces[0], agent_centroid_m, scene_index,agent_from_world)

        crosswalk_polylines = self.make_cross_polylines(agent_centroid_m,agent_from_world,len(dataset.scenes)!=1)

        if self.ignore_ego and len(dataset.scenes)>1:
            ego_polyline=np.concatenate([goal_pos,np.array([1.0],dtype=np.float32)], axis=0)[None]
        else:
            history_coords_offset, history_yaws_offset, history_extents, history_availability = get_relative_poses(
                self.history_num_frames_max + 1, history_frames, selected_track_id, history_agents, agent_from_world,
                agent_yaw_rad
            )

            abs_time = history_frames["timestamp"][:3]

            rel_pos=history_coords_offset[0]

            rel_yaw = history_yaws_offset[0]

            rel_acc = np.zeros([3],dtype=np.float32)

            if len(abs_time)>1:
                cur_time_gap = (abs_time[0] - abs_time[1]) / 1e9

                rel_vel=(rel_pos-history_coords_offset[1])/cur_time_gap

                angular_velocity = angular_distance(rel_yaw, history_yaws_offset[1]) / cur_time_gap

                if len(abs_time) > 2:
                    prev_time_gap = (abs_time[1] - abs_time[2]) / 1e9

                    prev_velocity=(history_coords_offset[1]-history_coords_offset[2])/prev_time_gap

                    rel_acc=(rel_vel-prev_velocity)/cur_time_gap

                    prev_angular_velocity=angular_distance(history_yaws_offset[1],history_yaws_offset[2])/prev_time_gap

                    angular_acceleration =(angular_velocity-prev_angular_velocity)/cur_time_gap

                    rel_acc = np.concatenate([rel_acc, angular_acceleration])

                rel_vel = np.concatenate([rel_vel, angular_velocity])

            else:
                rel_vel = np.zeros([3],dtype=np.float32)

            rel_pos = np.concatenate([rel_pos, rel_yaw])

            ego_polyline=np.concatenate([rel_pos,rel_vel,rel_acc ,goal_pos,np.array([1.0],dtype=np.float32)], axis=0)[None]

        real_time = future_frames["timestamp"]

        cur_time = history_frames["timestamp"][0]

        if len(dataset.scenes) > 1:

            pred_time = np.arange(1, self.future_num_frames + 1) * int(1e8) + cur_time

            x = np.interp(pred_time, real_time, future_coords_offset[:len(real_time), 0])
            y = np.interp(pred_time, real_time, future_coords_offset[:len(real_time), 1])

            yaw = np.interp(pred_time, real_time, future_yaws_offset[:len(real_time), 0])

            future_yaws_offset = yaw[:, None]

            future_coords_offset = np.stack([x, y], axis=-1)

            return lane_polylines,crosswalk_polylines,agent_polylines,ego_polyline,future_coords_offset,future_yaws_offset,future_availability
        else:

            data = {
                "lane_polylines": lane_polylines,
                "crosswalk_polylines": crosswalk_polylines,
                "agent_polylines": agent_polylines,
                "ego_polyline": ego_polyline,
                "target_positions": future_coords_offset,
                "target_yaws": future_yaws_offset,
                "target_availabilities": future_availability,

                "world_from_agent": world_from_agent,
                "centroid": agent_centroid_m,
                "yaw": agent_yaw_rad,
                "track_id": -1
            }

            if len(real_time)>0:
                data["next_timestep"]=real_time[0]-cur_time

            return data


    def make_cross_polylines(self, agent_centroid_m,agent_from_world,training):

        crosswalk_polylines=np.zeros([ self.max_num_crosswalks,self.max_points_per_crosswalk,self.cross_feat_dim])

        cross_indices = self.meta_manager.crosstree.query_ball_point(agent_centroid_m, self.max_retrieval_distance)

        cross_features = self.meta_manager.crosswalk_features[cross_indices]

        cross_ids = cross_features[:, -1]

        unique_ids = np.unique(cross_ids)

        for idx, cross_id in enumerate(unique_ids):
            cross_feature = self.meta_manager.crosswalk_features[self.meta_manager.crosswalk_features[:, -1] == cross_id]

            if training==False:

                cross_order=cross_feature[:,-2]

                roll_len=np.random.randint(low=1,high=len(cross_order))

                cross_feature[:,-2]=np.roll(cross_order,roll_len)

            crosswalk_polylines[idx,:len(cross_feature)]=cross_feature

        cross_num=len(unique_ids)

        crosswalk_polylines[:cross_num,:, :2] = transform_points(crosswalk_polylines[:cross_num,:, :2], agent_from_world)

        crosswalk_polylines[:cross_num,:, 2:4] = transform_points(crosswalk_polylines[:cross_num,:, 2:4], agent_from_world)

        return crosswalk_polylines.astype(np.float32)

    def make_lane_polylines(self, tl_faces, agent_centroid_m, scene_index,agent_from_world):

        lane_polylines = np.zeros([self.max_num_lanes, self.max_points_per_lane, self.lane_feat_dim])

        lane_indices = self.meta_manager.lanetree.query_ball_point(agent_centroid_m, self.max_retrieval_distance)

        if len(lane_indices):

            active_tl_faces = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())
            active_tl_face_to_color: Dict[str, str] = {}

            for face in active_tl_faces:
                try:
                    color = self.meta_manager.map_api.get_color_for_face(face)

                    active_tl_face_to_color[face] = color.lower()
                except KeyError:
                    continue  # this happens only on KIRBY, 2 TLs have no match in the map

            lane_ids = self.meta_manager.vector_laneids[lane_indices]

            unique_ids = np.unique(lane_ids)

            lane_features = self.meta_manager.vector_features[lane_indices]

            if self.dist_enhanced:
                lane_features[:, -4] = self.meta_manager.dist_graph[lane_indices, self.target_index[scene_index]]

                lane_cur_dists = self.meta_manager.cur_lane_dist(self.meta_manager.lanetree,self.meta_manager.dist_graph,agent_centroid_m, lane_indices)

                lane_cur_dist_min=np.zeros([len(unique_ids)])

                for i,lane_id in enumerate(unique_ids):
                    lane_cur_d = lane_cur_dists[lane_ids == lane_id]

                    lane_cur_min = lane_cur_d.min()

                    lane_cur_dist_min[i]=lane_cur_min

                connected_ids = unique_ids[lane_cur_dist_min < self.max_retrieval_distance]

                if len(connected_ids)>self.max_num_lanes:
                    unique_ids = unique_ids[np.argsort(lane_cur_dist_min)]
                else:
                    unique_ids =connected_ids

            for idx, lane_id in enumerate(unique_ids[:self.max_num_lanes]):

                lane_feature = lane_features[lane_ids == lane_id]

                lane_feature=lane_feature[:self.max_points_per_lane]

                lane_feature[:, -3] = self.meta_manager.map_api.get_tl_feature_for_lane(lane_id, active_tl_face_to_color)

                lane_polylines[idx][:len(lane_feature)]=lane_feature

            lane_num = len(unique_ids)

            lane_polylines[:lane_num, :,:2] = transform_points(lane_polylines[:lane_num,:, :2], agent_from_world)

            lane_polylines[:lane_num, :, 2:4] = transform_points(lane_polylines[:lane_num,:, 2:4], agent_from_world)

        return lane_polylines.astype(np.float32)

    def make_agent_polylines(self, centroid,agent_yaw_rad,agent_from_world,history_frames,history_agents):

        cur_agents = history_agents[0]

        cur_agents = filter_agents_by_distance(cur_agents, centroid, self.max_agents_distance)

        list_agents_to_take = cur_agents["track_id"]

        agent_polylines =np.zeros([ self.other_agents_num, self.agent_all_feat_dim], np.float32)

        abs_time=history_frames["timestamp"]

        abs_time=np.concatenate([abs_time,np.zeros([self.history_num_frames_agents+1-len(abs_time)])])

        rel_time=(abs_time[1:]-abs_time[0])/1e9

        for idx, track_id in enumerate(list_agents_to_take[:self.other_agents_num]):
            (
                agent_history_coords_offset,
                agent_history_yaws_offset,
                agent_history_extent,
                agent_history_availability,
            ) = get_relative_poses(self.history_num_frames_agents + 1, history_frames, track_id, history_agents,
                                   agent_from_world, agent_yaw_rad)

            agents_type_hist=np.zeros([self.history_num_frames_agents+1])
            for t in range(len(history_agents)):
                hist_agent=history_agents[-t]
                current_other_actor = hist_agent[hist_agent["track_id"] == track_id]
                if len(current_other_actor)>0:
                    agents_type =  np.array([np.argmax(current_other_actor["label_probabilities"])])+1

                    agents_type_hist[t]=agents_type

            rel_time_avil=rel_time*agent_history_availability[1:self.history_num_frames_agents+1]

            agent_polyline = np.concatenate(
                [agent_history_coords_offset[:self.history_num_frames_agents+1].reshape(-1), agent_history_yaws_offset[:self.history_num_frames_agents+1].reshape(-1),
                 agent_history_extent[:self.history_num_frames_agents+1].reshape(-1), agents_type_hist, rel_time_avil,
                  np.array([1])], axis=0)
            agent_polylines[idx]=agent_polyline

        return agent_polylines

    def get_scene_dataset(self, scene_index: int) :
        dataset = self.dataset.get_scene_dataset(scene_index)

        return LyftDataset(self.cfg, 'val', self.meta_manager, dataset, True)

