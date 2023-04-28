import torch
import numpy as np
from l5kit.data import get_agents_slice_from_frames
from l5kit.evaluation import metrics as l5metrics
from l5kit.planning import utils
from shapely.geometry import Point


class MyClosedLoopEvaluator:

    def __init__(self, data_type,visualizer,verbose=False):
        self.scene_fraction=0.8

        self.max_distance_ref_trajectory=2

        self.reset()

        self.verbose=verbose

        self.max_acc=3

        self.step_time=0.1

        self.visualizer = visualizer

        if data_type=="lyft":
            self.FRONT = 0
            self.REAR = 1
            self.SIDE = 2

            self.eval_start_frame=2

            EGO_EXTENT_WIDTH = 1.85
            EGO_EXTENT_LENGTH = 4.87

            self.simulated_extent = np.r_[EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH]


        else:
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

            map_root = "/mnt/workspace/nuplan/dataset/maps"

            map_version = 'nuplan-maps-v1.0'

            map_name="us-nv-las-vegas-strip"

            self.map_api = get_maps_api(map_root, map_version, map_name)

            self._max_violation_threshold=0.3

            self._driving_direction_violation_threshold=1

            self._time_horizon=1

    def reset(self):
        self.collision_front=0
        self.collision_rear=0
        self.collision_side=0
        self.collision_num=0

        self.off_road=0
        self.off_ref=0

        self.jerk=[]
        self.acc=[]

        self.scene_num=1e-5

        self.l2=[]
        self.dist =[]

    def validate(self):
        res_agg= {"collision_front": self.collision_front / self.scene_num,
                  "collision_rear": self.collision_rear / self.scene_num,
                  "collision_side": self.collision_side / self.scene_num,
                  "off_road": self.off_road / self.scene_num,
                  "off_ref":self.off_ref/self.scene_num
                  }

        res_agg["collision_rate"] = self.collision_num/self.scene_num
        res_agg["distance_ref_trajectory"] = torch.cat(self.dist, dim=0).mean()
        res_agg["displacement_error_l2"] =torch.cat(self.l2, dim=0).mean()


        acc_all=torch.cat(self.acc, dim=0)
        res_agg["acc_mean"]=acc_all.mean()
        res_agg["jerk_mean"]=torch.cat(self.jerk, dim=0).mean()

        res_agg["comfort"]=(acc_all>self.max_acc).to(float).mean()

        self.reset()

        return res_agg

    def common_metric(self,simulated_centroid,observed_ego_states,scene_name):

        velocity = (simulated_centroid[1:] - simulated_centroid[:-1]) / self.step_time

        acceleration = (velocity[1:] - velocity[:-1]) / self.step_time

        self.acc.append(torch.norm(acceleration, p=2, dim=-1))

        jerk = (acceleration[1:] - acceleration[:-1]) / self.step_time

        self.jerk.append(torch.norm(jerk, p=2, dim=-1))

        l2 = torch.norm(simulated_centroid -  observed_ego_states[:len(simulated_centroid)], p=2, dim=-1)

        self.l2.append(l2)

        # Trim the simulated trajectory to have a specified fraction
        simulated_fraction_length = int(len(simulated_centroid) * self.scene_fraction)
        simulated_centroid_fraction = simulated_centroid[:simulated_fraction_length]

        lat_distance = l5metrics.distance_to_reference_trajectory(simulated_centroid_fraction,  observed_ego_states.unsqueeze(0))

        off_ref = torch.any(lat_distance > self.max_distance_ref_trajectory)

        self.off_ref += off_ref

        if self.verbose and off_ref:
            print("off_ref",scene_name)

        self.dist.append(lat_distance)

    def evaluate(self, simulation_outputs,data_type):
        for simulation_output in simulation_outputs:
            self.scene_num += 1

            if "nuplan" in data_type:
                self.compute_nuplan_metric(simulation_output)
            else:
                self.compute_metric(simulation_output)

                if self.visualizer is not None:
                    from l5kit.visualization.visualizer.visualizer import visualize
                    from bokeh.io import show

                    vis_in = self.visualizer.lyft_visualizer_scene(simulation_output)
                    layout = visualize(simulation_output.scene_id, vis_in)

                    show(layout)

    def compute_off_road(self,histories,scene_id):
        from nuplan.planning.metrics.utils.route_extractor import extract_corners_route,get_route,get_distance_of_closest_baseline_point_to_its_start
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        ego_footprint_list = [history[0].car_footprint for history in histories]

        corners_lane_lane_connector_list = extract_corners_route(self.map_api, ego_footprint_list)

        all_ego_corners =[history[0].car_footprint.all_corners() for history in histories]

        ego_poses = [history[0].center.point for history in histories]

        ego_driven_route = get_route(self.map_api, ego_poses)

        for ego_corners, corners_lane_lane_connector,center_lane_lane_connector in zip(
            all_ego_corners, corners_lane_lane_connector_list,ego_driven_route
        ):
            for ind, route_object in enumerate(corners_lane_lane_connector):
                ego_corner = ego_corners[ind]

                if not route_object and not self.map_api.is_in_layer(ego_corner,  layer=SemanticMapLayer.DRIVABLE_AREA):

                    if center_lane_lane_connector:
                        distance = float(min(obj.polygon.distance(Point(*ego_corner)) for obj in center_lane_lane_connector))   #self.compute_distance_to_map_objects_list(ego_corner, center_lane_lane_connector)

                        if distance > self._max_violation_threshold:
                            id_distance_tuple = self.map_api.get_distance_to_nearest_map_object(ego_corner,
                                                                                                layer=SemanticMapLayer.DRIVABLE_AREA)

                            if id_distance_tuple[1] is None or id_distance_tuple[1] >= self._max_violation_threshold:
                                self.off_road += 1

                                if self.verbose:
                                    print('off road', scene_id)

                                return

    def compute_nuplan_metric(self,simulation_output):
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.state_representation import StateSE2

        histories,expert_states,scene_id=simulation_output

        simulated_centroid = torch.stack([torch.tensor(history[0].rear_axle.array) for history in histories],
                                         dim=0)

        self.common_metric(simulated_centroid, expert_states, scene_id)

        self.compute_off_road(histories,scene_id)

        for frame_idx,history in enumerate(histories):
            if frame_idx==0:
                continue

            ego_state,target_agents=history

            pred_centroid=ego_state.center.array

            pred_extent=np.r_[ego_state.car_footprint.length, ego_state.car_footprint.width]

            target_agents_centroid=target_agents[:,:2]

            target_agents_extent=target_agents[:,3:5]       #lengthwidth,

            ego_bbox = ego_state.car_footprint.oriented_box.geometry

            within_range_mask = utils.within_range(pred_centroid, pred_extent,   target_agents_centroid, target_agents_extent)

            for agent in target_agents[within_range_mask]:

                pose=StateSE2(agent[0], agent[1], agent[2])

                agent_box=OrientedBox(pose, width=agent[3], length=agent[4], height=agent[5]).geometry#tracked_object.box

                if ego_bbox.intersects(agent_box):
                    self.collision_num += 1

                    if self.verbose:
                        print('col', scene_id, frame_idx)

                    return

    def compute_metric(self,simulation_output):

        simulated_scene_ego_state = simulation_output.simulated_ego_states

        simulated_centroid = simulated_scene_ego_state[:, :2]  # [Timesteps, 2]

        observed_ego_states = simulation_output.recorded_ego_states[:, :2]

        self.common_metric(simulated_centroid,observed_ego_states,simulation_output.scene_id)

        num_frames = simulated_scene_ego_state.size(0)
        simulated_agents = simulation_output.simulated_agents
        simulated_egos = simulation_output.simulated_ego

        for frame_idx in range(self.eval_start_frame,num_frames):
            simulated_ego_state_frame = simulated_scene_ego_state[frame_idx]
            simulated_ego_frame = simulated_egos[frame_idx]
            simulated_agent_frame = simulated_agents[get_agents_slice_from_frames(simulated_ego_frame)]

            simulated_centroid = simulated_ego_state_frame[:2].cpu().numpy()
            simulated_angle = simulated_ego_state_frame[2].cpu().numpy()
            result = l5metrics.detect_collision(simulated_centroid, simulated_angle,
                                                       self.simulated_extent, simulated_agent_frame)

            if result is not None:
                collision_type=result[0]

                if collision_type==self.FRONT:
                    self.collision_front+=1
                elif collision_type==self.REAR:
                    self.collision_rear+=1
                else:
                    self.collision_side+=1

                self.collision_num+=1
                if self.verbose:

                    print('col',simulation_output.scene_id,frame_idx)

                break




