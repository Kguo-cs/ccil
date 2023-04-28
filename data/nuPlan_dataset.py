import os
from torch.utils.data import Dataset

import numpy as np
from l5kit.geometry import angular_distance, compute_agent_pose,rotation33_as_yaw,yaw_as_rotation33
from l5kit.geometry.transform import transform_points,transform_point
from .nuPlan_load import db_process


class nuPlanDataset(Dataset):

    def __init__(self,cfg,type,meta_manager):

        model_cfg = cfg['model_params']

        data_cfg = cfg['data_generation_params']

        map_cfg = data_cfg["lane_params"]

        self.step_time = model_cfg['step_time']

        self.causal_len = model_cfg['causal_len']
        self.causal_interval = model_cfg['causal_interval']

        self.history_num_frames_agents = model_cfg["history_num_frames_agents"]
        self.future_num_frames = model_cfg["future_num_frames"]
        self.lane_feat_dim = model_cfg["lane_feat_dim"] + 1
        self.cross_feat_dim = model_cfg["cross_feat_dim"] + 1
        self.ego_feat_dim = model_cfg["ego_feat_dim"] + 1
        self.act_dim = model_cfg["act_dim"]
        self.agent_feat_dim = model_cfg["agent_feat_dim"]
        self.agent_all_feat_dim = self.agent_feat_dim * (self.history_num_frames_agents + 1) + 1

        self.dist_enhanced=data_cfg["dist_enhanced"]
        self.centroid_std = data_cfg["centroid_std"]
        self.other_agents_num = data_cfg["other_agents_num"]
        self.max_agents_distance = data_cfg["max_agents_distance"]

        self.max_points_per_lane = map_cfg["max_points_per_lane"]
        self.max_points_per_crosswalk = map_cfg["max_points_per_crosswalk"]
        self.max_retrieval_distance = map_cfg["max_retrieval_distance_m"]
        self.max_num_lanes = map_cfg["max_num_lanes"]
        self.max_num_crosswalks = map_cfg["max_num_crosswalks"]

        self.sim_len=250

        self.future_horizon=self.future_num_frames*self.step_time

        self.history_horizon_agents=self.history_num_frames_agents*self.step_time

        self.meta_manager = meta_manager

        try:
            data=np.load('./data/nuplan_meta/'+type+".npz",allow_pickle=True)

            self.scenarios=data["scenarios"][::self.sim_len]

            self.ego_array=data["ego_array"]
            self.agent_array = data["agent_array"]
            self.red_tl_array=data["red_tl_array"]
            self.green_tl_array=data["green_tl_array"]


            self.agent_index=data["agent_index"]
            self.tl_red_index=data["tl_red_index"]
            self.tl_green_index=data["tl_green_index"]

            self.time_index = data["time_index"]

            self.scenarios_info=data["scenarios_info"]

        except:
            data_root = os.environ["NUPLAN_DATA_FOLDER"]+'nuplan-v1.0/'+type

            ego_array,agent_array,red_tl_array,green_tl_array,scenarios,scenarios_info=db_process(type,data_root)

            scenarios_info=scenarios_info[::self.sim_len]

            self.scenarios_info=meta_manager.get_scenetargets(ego_array, scenarios_info,goal_num=20)

            self.agent_index = ego_array[:, 3].astype(np.uint32)
            self.tl_red_index =  ego_array[:,4].astype(np.uint32)
            self.tl_green_index =ego_array[:,5].astype(np.uint32)
            self.time_index =ego_array[:,6].astype(np.uint64)

            self.ego_array=ego_array[:,:3]
            self.agent_array=agent_array
            self.red_tl_array = red_tl_array
            self.green_tl_array = green_tl_array

            np.savez("./data/nuplan_meta/"+type ,
                             ego_array = self.ego_array,
                             agent_array=self.agent_array,
                             red_tl_array=self.red_tl_array,
                             green_tl_array=self.green_tl_array,
                             agent_index=self.agent_index,
                             tl_red_index = self.tl_red_index,
                             tl_green_index = self.tl_green_index,
                             time_index=self.time_index,
                             scenarios = scenarios,
                             scenarios_info = self.scenarios_info)

        self.scene_num = len(self.scenarios_info)

        print("scene_num", self.scene_num)

    def __len__(self):
        return len(self.scenarios_info)*self.sim_len

    def __getitem__(self, index) -> dict:

        lane_polylines=[]
        crosswalk_polylines=[]
        agent_polylines=[]
        ego_polyline=[]

        target_positions = np.zeros([self.causal_len, self.future_num_frames,2], np.float32)
        target_yaws = np.zeros([self.causal_len, self.future_num_frames,1], np.float32)
        target_availabilities = np.ones([self.causal_len, self.future_num_frames], np.bool)

        scene_index=index//self.sim_len

        state_index=index%self.sim_len

        scene_info = self.scenarios_info[scene_index]

        for t in range(self.causal_len):
            frame_index = state_index + (-self.causal_len + 1 + t) * self.causal_interval

            lane_polylines_t,crosswalk_polylines_t,agent_polylines_t,ego_polyline_t,future_coords_offset,future_yaws_offset = self.get_frame(
                frame_index, scene_info)

            lane_polylines.append(lane_polylines_t)
            crosswalk_polylines.append(crosswalk_polylines_t)
            agent_polylines.append(agent_polylines_t)
            ego_polyline.append(ego_polyline_t)

            target_positions[t] = future_coords_offset
            target_yaws[t] = future_yaws_offset

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

    def get_frame(self,frame_index, info ,cur_ego_states=None):

        start_time=info[0].astype(int)
        goal_translation=info[1:3]
        target_index=info[3].astype(int)

        cur_time = start_time + frame_index

        if cur_ego_states is None:
            ego_centroid_m = self.ego_array[cur_time][:2]+ np.random.randn(2) * self.centroid_std
        else:
            ego_centroid_m = cur_ego_states.rear_axle.array

        vector = goal_translation - ego_centroid_m

        agent_yaw_rad = np.arctan2(vector[1], vector[0])

        world_from_agent = compute_agent_pose(ego_centroid_m, agent_yaw_rad)

        agent_from_world = np.linalg.inv(world_from_agent)

        goal_pos = transform_point(goal_translation, agent_from_world)

        agent_polylines,all_agents=self.make_agent_polylines(cur_time,agent_from_world, ego_centroid_m,agent_yaw_rad)

        lane_polylines = self.make_lane_polylines( ego_centroid_m,cur_time, target_index,agent_from_world)

        crosswalk_polylines = self.make_cross_polylines(ego_centroid_m,agent_from_world,cur_ego_states)

        if cur_ego_states is None:
            ego_polyline=np.concatenate([goal_pos,np.array([1.0])], axis=0)[None].astype(np.float32)

            cur_fut=self.ego_array[cur_time:cur_time+self.future_num_frames+2,:3]

            real_time=self.time_index[cur_time:cur_time+self.future_num_frames+2]

            pred_time=np.arange(1,self.future_num_frames+1)*int(1e5)+real_time[0]

            x=np.interp(pred_time, real_time, cur_fut[:,0])
            y=np.interp(pred_time, real_time, cur_fut[:,1])
            yaw=np.interp(pred_time, real_time, cur_fut[:,2])

            pos=np.stack([x,y],axis=-1)
            # pos=self.ego_array[cur_time+1:cur_time+self.future_num_frames+1,:2]
            # yaw=self.ego_array[cur_time+1:cur_time+self.future_num_frames+1,2]

            future_coords_offset = transform_points(pos, agent_from_world)

            future_yaws_offset = angular_distance(yaw[:,None], agent_yaw_rad)

            return lane_polylines,crosswalk_polylines,agent_polylines,ego_polyline,future_coords_offset,future_yaws_offset

        else:
            cur_vel = cur_ego_states.dynamic_car_state.rear_axle_velocity_2d.array

            cur_acc = cur_ego_states.dynamic_car_state.rear_axle_acceleration_2d.array

            rel_pos = transform_point(cur_ego_states.rear_axle.array,agent_from_world)

            transf_matrix = np.transpose(agent_from_world, (1, 0))

            rel_vel=cur_vel @ transf_matrix[:2, :2]

            rel_acc=cur_acc @ transf_matrix[:2, :2]

            rel_yaw= angular_distance(cur_ego_states.rear_axle.heading,agent_yaw_rad)

            rel_pos=np.concatenate([rel_pos,np.array([rel_yaw])])

            rel_vel=np.concatenate([rel_vel,np.array([cur_ego_states.dynamic_car_state.angular_velocity])])

            rel_acc=np.concatenate([rel_acc,np.array([cur_ego_states.dynamic_car_state.angular_acceleration])])

            ego_polyline=np.concatenate([rel_pos, rel_vel ,rel_acc,goal_pos,np.array([1.0])], axis=0)[None].astype(np.float32)#,    np.array([yaw]) , cur_speed

            return {
                "lane_polylines": lane_polylines,
                "crosswalk_polylines": crosswalk_polylines,
                "agent_polylines": agent_polylines,
                "ego_polyline": ego_polyline,
                "world_from_agent": world_from_agent,
                "yaw": agent_yaw_rad,
                }\
                ,all_agents[:,:6]

    def make_lane_polylines(self,agent_centroid_m,cur_time,target_index,agent_from_world):

        lane_polylines = np.zeros([self.max_num_lanes, self.max_points_per_lane, self.lane_feat_dim])

        lane_indices = self.meta_manager.lanetree.query_ball_point(agent_centroid_m, self.max_retrieval_distance)

        if len(lane_indices):

            red_lane_connectors=self.red_tl_array[self.tl_red_index[cur_time-1]: self.tl_red_index[cur_time] ]

            green_lane_connectors = self.green_tl_array[self.tl_green_index[cur_time-1]:self.tl_green_index[cur_time]]

            lane_features = self.meta_manager.vector_features[lane_indices]

            lane_ids = lane_features[:,-1].astype(np.uint32)

            unique_ids = np.unique(lane_ids)

            if self.dist_enhanced:

                lane_features[:, -4] = self.meta_manager.dist_graph[lane_indices, target_index]

                lane_cur_dists = self.meta_manager.cur_lane_dist(self.meta_manager.lanetree,self.meta_manager.dist_graph, agent_centroid_m, lane_indices)

                lane_cur_dist_min=np.zeros([len(unique_ids)])

                for i,lane_id in enumerate(unique_ids):
                    lane_cur_dist_min[i]= lane_cur_dists[lane_ids == lane_id].min()

                connected_ids = unique_ids[lane_cur_dist_min < self.max_retrieval_distance]

                if len(connected_ids)>self.max_num_lanes:
                    unique_ids = unique_ids[np.argsort(lane_cur_dist_min)]
                else:
                    unique_ids =connected_ids

            for idx, lane_id in enumerate(unique_ids[:self.max_num_lanes]):

                lane_feature = lane_features[lane_ids == lane_id]

                lane_feature=lane_feature[:self.max_points_per_lane]

                if lane_id in green_lane_connectors:
                    lane_feature[:, -3] =1
                elif lane_id in red_lane_connectors:
                    lane_feature[:,-3] =-1

                lane_polylines[idx][:len(lane_feature)]=lane_feature

            lane_num = len(unique_ids)

            lane_polylines[:lane_num, :,:2] = transform_points(lane_polylines[:lane_num,:, :2], agent_from_world)

            lane_polylines[:lane_num, :, 2:4] = transform_points(lane_polylines[:lane_num,:, 2:4], agent_from_world)

        return lane_polylines.astype(np.float32)

    def make_cross_polylines(self,agent_centroid_m,agent_from_world,cur_ego_states):

        crosswalk_polylines=np.zeros([ self.max_num_crosswalks,self.max_points_per_crosswalk,self.cross_feat_dim])

        cross_indices = self.meta_manager.crosstree.query_ball_point(agent_centroid_m, self.max_retrieval_distance)

        crosswalk_features=self.meta_manager.crosswalk_features

        cross_features = crosswalk_features[cross_indices]

        cross_ids = cross_features[:, -1]

        unique_ids=np.unique(cross_ids)[:self.max_num_crosswalks]#according to importance

        for idx, cross_id in enumerate(unique_ids):
            cross_feature = crosswalk_features[crosswalk_features[:, -1] == cross_id]

            if cur_ego_states is None:

                cross_order=cross_feature[:,-2]

                roll_len=np.random.randint(low=1,high=len(cross_order))

                cross_feature[:,-2]=np.roll(cross_order,roll_len)

            crosswalk_polylines[idx]=cross_feature

        cross_num=len(unique_ids)

        crosswalk_polylines[:cross_num,:, :2] = transform_points(crosswalk_polylines[:cross_num,:, :2], agent_from_world)

        crosswalk_polylines[:cross_num,:, 2:4] = transform_points(crosswalk_polylines[:cross_num,:, 2:4], agent_from_world)

        return crosswalk_polylines.astype(np.float32)

    def make_agent_polylines(self,  cur_time,agent_from_world, ego_centroid_m,agent_yaw_rad):
        agent_polylines = np.zeros([self.other_agents_num, self.history_num_frames_agents + 1, self.agent_feat_dim])

        all_agents=self.agent_array[self.agent_index[cur_time-1]:self.agent_index[cur_time]]

        agents_dist = np.linalg.norm(ego_centroid_m - all_agents[:, :2], ord=2, axis=-1)

        agent_nearby=all_agents[agents_dist<self.max_agents_distance]

        agents_dist = agents_dist[agents_dist < self.max_agents_distance]

        nearby_agent_sorted = agent_nearby[np.argsort(agents_dist)[:self.other_agents_num]]

        list_agents_to_take=nearby_agent_sorted[:,-1]

        rel_time=(self.time_index[cur_time]-self.time_index[cur_time-self.history_num_frames_agents:cur_time])/1e6

        for t in range(self.history_num_frames_agents):

            tracks=self.agent_array[self.agent_index[cur_time-t-2]:self.agent_index[cur_time-t-1]]

            for track in tracks:

                idx = np.where(list_agents_to_take == track[-1])[0]

                if len(idx) > 0:
                    i = idx[0]

                    agent_polylines[i][t][:-1] = track[:-1]
                    agent_polylines[i][t][-1] = rel_time[-t-1]

        agent_polylines[:len(nearby_agent_sorted),-1]=nearby_agent_sorted

        agent_polylines[:, :, :2] = transform_points(agent_polylines[:, :, :2], agent_from_world)

        agent_polylines[:, :, 2] = angular_distance(agent_polylines[:, :, 2], agent_yaw_rad)

        transf_matrix = np.expand_dims(agent_from_world, 0)

        transf_matrix = np.transpose(transf_matrix, (0, 2, 1))

        agent_polylines[:,:,6:8] = agent_polylines[:,:,6:8] @ transf_matrix[:, :2, :2]

        availability=agent_polylines[:,:,-1:].astype(np.bool)

        agent_polylines=agent_polylines*availability

        agent_polylines=agent_polylines.reshape(self.other_agents_num,-1)

        return agent_polylines.astype(np.float32),all_agents

