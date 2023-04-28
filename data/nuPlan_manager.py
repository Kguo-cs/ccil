import numpy as np
from tqdm import tqdm
from l5kit.data.map_api import InterpolationMethod
from scipy.spatial import KDTree
from nuplan.common.maps.nuplan_map.utils import get_all_rows_with_value, get_row_with_value
from .map_manager import MapManager
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
import os

def xy_interpolate(xy: np.ndarray, step: float, method: InterpolationMethod):
    cum_dist = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))
    cum_dist = np.insert(cum_dist, 0, 0)

    if method == InterpolationMethod.INTER_ENSURE_LEN:
        step = int(step)
        assert step > 1, "step must be at least 2 with INTER_ENSURE_LEN"
        steps = np.linspace(cum_dist[0], cum_dist[-1], step)

    elif method == InterpolationMethod.INTER_METER:
        assert step > 0, "step must be greater than 0 with INTER_FIXED"
        steps = np.arange(cum_dist[0], cum_dist[-1], step)
    else:
        raise NotImplementedError(f"interpolation method should be in {InterpolationMethod.__members__}")

    xy_inter = np.empty((len(steps), 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=xy[:, 1])

    return xy_inter

class nuPlanMapManager(MapManager):
    def __init__(self,cfg):

        try:
            print("load map data")

            vectormap = np.load("./data/nuplan_meta/nuplan_map_features.npz")

            if cfg["data_generation_params"]["dist_enhanced"]:
                self.dist_graph = vectormap['dist_graph']

            self.vector_features = vectormap["vector_features"]

            self.crosswalk_features = vectormap["crosswalk_features"]

            print("load map, done")

        except:
            map_root = os.environ["NUPLAN_DATA_FOLDER"] + "/maps"

            map_version = 'nuplan-maps-v1.0'

            map_name="us-nv-las-vegas-strip"

            self.map_api = get_maps_api(map_root, map_version, map_name)

            self.dist_graph, self.vector_features, self.crosswalk_features = self.build_graphs()

            np.savez_compressed('./data/nuplan_meta/nuplan_map_features', dist_graph=self.dist_graph,vector_features=self.vector_features,crosswalk_features=self.crosswalk_features)

            print("build all maps done")

        self.lanetree = KDTree(self.vector_features[:,:2])

        self.crosstree = KDTree(self.crosswalk_features[:, :2])

    def get_scenetargets(self , cur_postions, scenerio_information,goal_num=20):
        print("build goal dist")

        scene_target = scenerio_information[:,1:3]

        unique_targets = np.unique(scene_target, axis=0)#9363        #for i,goal_pos in enumerate(scene_target):

        target_index=np.zeros_like(scenerio_information[:,-1])

        for goal_pos in tqdm(unique_targets):

            frame_index=cur_postions[:,-2:]==goal_pos

            frame_index=frame_index[:,0]&frame_index[:,1]

            frames=cur_postions[frame_index,:2]

            scene_index=scene_target==goal_pos

            scene_index=scene_index[:,0]&scene_index[:,1]

            dist_final, final_indices = self.lanetree.query(goal_pos, goal_num)

            scene_len = len(frames)

            final_dist_graph = self.dist_graph[:, final_indices]

            scene_goal_dist_min = np.zeros([scene_len, goal_num])

            for index in range(scene_len):

                scene_goal_dist_min[index] = self.min_dist_to_goal(self.vector_features,self.lanetree,frames[index],final_dist_graph)

          #  min_dist=np.max(scene_goal_dist_min,axis=1)

            mean_dist = np.mean(scene_goal_dist_min, axis=0)

            weighted_dist = dist_final * 5 + mean_dist

            min_mean = np.argmin(weighted_dist)

            goal_index = final_indices[min_mean]

            target_index[scene_index]=goal_index

        target_scenerio_information=np.concatenate([scenerio_information,target_index[:,None]],axis=-1)

        return target_scenerio_information.astype(np.float32)

    def build_crosswalk(self):

        crosswalk_features = []

        idx=0

        for layer_i,layer_name in enumerate(["stop_polygons",'crosswalks','intersections','walkways', 'carpark_areas']):#

            layer_df=self.map_api._load_vector_map_layer(layer_name)

            obj_ids = layer_df["fid"].tolist()

            for id in obj_ids:

                obj = get_row_with_value(layer_df, "fid", id)

                x_coords, y_coords=obj.geometry.boundary.coords.xy

                points = xy_interpolate(np.stack([x_coords, y_coords], axis=1), 21, InterpolationMethod.INTER_ENSURE_LEN)#np.stack([x_coords, y_coords], axis=1)#

                next_point = points[1:]

                feature = np.concatenate([points[:-1, :2], next_point[:, :2],np.zeros([len(next_point), 1]) + layer_i, np.arange(len(next_point))[:, None], np.ones([len(next_point), 1]) + idx], axis=1)

                crosswalk_features.extend(feature)

                idx+=1

        return np.array(crosswalk_features)

    def build_vectors(self,lane,point_dist=3):

        x_coords, y_coords=lane.geometry.coords.xy

        vectors = xy_interpolate(np.stack([x_coords, y_coords], axis=1), point_dist, InterpolationMethod.INTER_METER)

        if len(vectors) < 2:
            vectors = xy_interpolate(np.stack([x_coords, y_coords], axis=1), 2, InterpolationMethod.INTER_ENSURE_LEN)

        return vectors


    def build_lane_graph(self):
        lane_graph = {}

        _lanes_df = self.map_api._load_vector_map_layer('lanes_polygons')
        _baseline_paths_df = self.map_api._load_vector_map_layer('baseline_paths')
        _boundaries_df = self.map_api._load_vector_map_layer('boundaries')
        _lane_connectors_df = self.map_api._load_vector_map_layer('lane_connectors')

        lane_ids = _lanes_df["fid"].tolist()

        for id in lane_ids:

            lane = get_row_with_value(_lanes_df, "fid", id)

            lane_group_fid = lane["lane_group_fid"]

            all_lanes = get_all_rows_with_value(_lanes_df, "lane_group_fid", lane_group_fid)

            lane_index = lane["lane_index"]

            # According to the map attributes, lanes are numbered left to right with smaller indices being on the left and larger indices being on the right
            left_lane_id = all_lanes[all_lanes["lane_index"] == int(lane_index) - 1]['fid']#.item()

            if left_lane_id.empty:
                adj_left=''
            else:
                adj_left=left_lane_id.item()

            right_lane_id = all_lanes[all_lanes["lane_index"] == int(lane_index) + 1]['fid']#.item()

            if right_lane_id.empty:
                adj_right=''
            else:
                adj_right=right_lane_id.item()

            childrens = get_all_rows_with_value(_lane_connectors_df, "exit_lane_fid", id)["fid"].to_list()

            baseline = get_row_with_value(_baseline_paths_df, "lane_fid", id)

            mids =self.build_vectors(baseline)

            left_boundary = get_row_with_value(_boundaries_df, "fid",    str(lane["left_boundary_fid"]))

            lefts =np.stack(left_boundary.geometry.coords.xy,axis=-1)

            right_boundary = get_row_with_value(_boundaries_df, "fid",  str(lane["right_boundary_fid"]))

            rights=np.stack(right_boundary.geometry.coords.xy,axis=-1)

            speed_limit = lane["speed_limit_mps"]
            if speed_limit is None or not speed_limit == speed_limit:
                speed_limit = 0

            lane_graph[id] = {
                'adj_left': adj_left,
                'adj_right': adj_right,
                'childrens': childrens,
                'mids': mids,
                'lefts': lefts,
                'rights': rights,
                "type": 0,
                "speed_limit": speed_limit
            }

        _lane_connector_polygon_df = self.map_api._load_vector_map_layer('gen_lane_connectors_scaled_width_polygons')

        lane_connector_ids = _lane_connectors_df["fid"].tolist()

        for id in lane_connector_ids:

            lane_connector = get_row_with_value(_lane_connectors_df, "fid", id)

            childrens = [str(lane_connector["entry_lane_fid"])]

            baseline = get_row_with_value(_baseline_paths_df, "lane_connector_fid", id)

            mids =self.build_vectors(baseline)

            lane_connector_polygon = get_row_with_value(_lane_connector_polygon_df, "lane_connector_fid", id)

            left_boundary = get_row_with_value(_boundaries_df, "fid",
                                                    str(lane_connector_polygon["left_boundary_fid"]))

            lefts =np.stack(left_boundary.geometry.coords.xy,axis=-1)

            right_boundary = get_row_with_value(_boundaries_df, "fid", str(
                lane_connector_polygon["right_boundary_fid"]))

            rights=np.stack(right_boundary.geometry.coords.xy,axis=-1)

            speed_limit = lane_connector["speed_limit_mps"]
            if speed_limit is None or not speed_limit == speed_limit:
                speed_limit = 0

#            lane_group_fid = lane_connector["lane_group_connector_fid"]

            lane_graph[id] = {
                'adj_left': '',
                'adj_right': '',
                'childrens': childrens,
                'mids': mids,
                'lefts': lefts,
                'rights': rights,
                "type": 1,
                "speed_limit":speed_limit
            }

        return lane_graph

    def build_vector_graph(self, lane_graph):
        vector_features = []
        vector_list = []
        vector_id_dict = {}

        for idx, (lane_id, v) in enumerate(lane_graph.items()):

            cur_mids = v['mids']

            left_lane_id = v['adj_left']
            right_lane_id = v['adj_right']

            lane_len = len(cur_mids)

            if len(left_lane_id) > 0:
                left_mids = lane_graph[left_lane_id]['mids']

                left_waypoint_ids, left_dists = self.connect_nearest(cur_mids, left_mids)

            else:
                left_waypoint_ids = [None] * lane_len
                left_dists = [0] * lane_len

            if len(right_lane_id) > 0:
                right_mids = lane_graph[right_lane_id]['mids']

                right_waypoint_ids, right_dists = self.connect_nearest(cur_mids, right_mids)
            else:
                right_waypoint_ids = [None] * lane_len
                right_dists = [0] * lane_len

            next_ids = [[lane_id + str(i)] for i in range(1, lane_len)]

            next_ids.append([child + str(0) for child in v['childrens']])

            diffs = np.diff(cur_mids, axis=0)

            next_dists = list(np.linalg.norm(diffs, axis=1))

            next_postions = []

            for pos in cur_mids[1:]:
                next_postions.append(pos)

            last_pos = cur_mids[-1]

            if len(v['childrens']) > 0:

                child_id = v['childrens'][0]

                last_next_pos = lane_graph[child_id]['mids'][0]
            else:
                last_next_pos = last_pos

            last_diff = last_next_pos - last_pos

            child_dist = np.linalg.norm(last_diff, axis=0)

            next_dists.append(child_dist)

            next_postions.append(last_next_pos)

            lane_left = v['lefts']
            lane_right = v['rights']

            for order, (mid, next_pos, left_id, left_dist, right_id, right_dist, next_id_, next_dist) in enumerate(
                    zip(cur_mids, next_postions, left_waypoint_ids, left_dists, right_waypoint_ids, right_dists,
                        next_ids, next_dists)):
                left_id = left_lane_id + str(left_id)
                right_id = right_lane_id + str(right_id)

                left_width = np.min(np.linalg.norm(lane_left - mid[None], axis=-1))
                right_width = np.min(np.linalg.norm(lane_right - mid[None], axis=-1))
                #width = min(left_width, right_width)
                #roadblock_id=v["roadblock_id"]

                dist_to_goal = 0
                tl_feature = 0

                type=v["type"]

                speed_limit=v["speed_limit"]

                features = [mid[0], mid[1], next_pos[0], next_pos[1], left_width, right_width, left_dist, right_dist,speed_limit,type, dist_to_goal, tl_feature, order,  lane_id]

                vector_features.append(features)

                vector_dict = {
                    'lane_id': lane_id,
                    'left_id': left_id,
                    'left_dist': left_dist,
                    'right_id': right_id,
                    'right_dist': right_dist,
                    'next_id': next_id_,
                    'next_dist': next_dist,
                }

                vector_id_dict[lane_id + str(order)] = len(vector_list)
                vector_list.append(vector_dict)

        return vector_list, vector_id_dict, np.array(vector_features).astype(np.float64)

    def build_graphs(self):
        print("build graph")

        lane_graph = self.build_lane_graph()  # 8505

        crosswalk_features = self.build_crosswalk()

        vector_list, vector_id_dict, vector_features = self.build_vector_graph(lane_graph)  # 67004

        nei_graph = self.build_neighbor_graph(vector_list, vector_id_dict)

        dist_graph = self.build_dist_graph(nei_graph)  # #np.load("/home/xuanyuan/PycharmProjects/myplan/dt/data/nuplan_meta/nuplan_map_features_64.npz")['dist_graph']#

        print("build graph, done")

        return dist_graph, vector_features,  crosswalk_features

