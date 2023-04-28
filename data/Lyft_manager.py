from l5kit.data.map_api import InterpolationMethod, MapAPI

import numpy as np
from scipy.spatial import KDTree
from .map_manager import MapManager
from tqdm import tqdm

class LyftManager(MapManager):
    def __init__(self, cfg, dm):

        self.map_api = MapAPI.from_cfg(dm, cfg)

        try:
            print("load map data")
            vectormap = np.load("./data/lyft_meta/lyft_map_features.npz", allow_pickle=True)

            if cfg["data_generation_params"]["dist_enhanced"]:
                self.dist_graph = vectormap['dist_graph']

            self.vector_features = vectormap["vector_features"]

            self.vector_laneids = vectormap["vector_laneids"]

            self.crosswalk_features = vectormap["crosswalk_features"]

            print("load map, done")

        except:

            self.dist_graph, self.vector_features, self.vector_laneids, self.crosswalk_features = self.build_graphs()

            np.savez_compressed('./data/lyft_meta/lyft_map_features',
                                vector_features=self.vector_features,
                                vector_laneids=self.vector_laneids,
                                dist_graph=self.dist_graph,
                                crosswalk_features=self.crosswalk_features)

        vector_start = self.vector_features[:, :2]

        self.lanetree = KDTree(vector_start)

        self.crosstree = KDTree(self.crosswalk_features[:, :2])

    def get_scenetargets(self ,type, frames, cumulative_sizes,goal_num=20):
        try:
            goal = np.load("./data/lyft_meta/" + type + "_lane_goal20.npy")

            scene_target=goal[:,1:]
            target_index=goal[:,0].astype(np.int)
        except:
            print("build goal dist")
            target_index=[]
            scene_target=[]

            for scene_index in tqdm(range(len(cumulative_sizes) - 1)):

                start_id = cumulative_sizes[scene_index]

                end_id = cumulative_sizes[scene_index + 1]

                last_frame = frames[end_id - 1]

                goal_pos = last_frame['ego_translation'][:2]

                dist_final, final_indices = self.lanetree.query(goal_pos, goal_num)

                scene_len = end_id - start_id

                dist_graph = self.dist_graph[:, final_indices]

                scene_goal_dist_min = np.zeros([scene_len, goal_num])

                for index in range(start_id, end_id):

                    cur_pos = frames[index]["ego_translation"][:2]

                    scene_goal_dist_min[index - start_id] = self.min_dist_to_goal(self.vector_features,self.lanetree,cur_pos,dist_graph)

                mean_dist = np.mean(scene_goal_dist_min, axis=0)

                weighted_dist = dist_final * 5 + mean_dist

                min_mean = np.argmin(weighted_dist)

                goal_index = final_indices[min_mean]

                target_lane_ids = self.vector_laneids[int(goal_index)]

                index = np.where(self.vector_laneids == target_lane_ids)[0]

                last_index = np.max(index)

                target_index.append(last_index)

                scene_target.append(self.vector_features[last_index, :2])

            scene_target = np.array(scene_target)

            target_index=np.array(target_index)

            goal_array=np.concatenate([target_index[:,None],scene_target],axis=-1)

            np.save("./data/lyft_meta/" + type + "_lane_goal20.npy",goal_array)

        return scene_target,target_index

    def build_crosswalk(self):

        crosswalks_ids = self.map_api.bounds_info["crosswalks"]["ids"]  # 8505

        crosswalk_features = []

        for idx, crosswalks_id in enumerate(crosswalks_ids):
            points = self.map_api.get_crosswalk_coords(crosswalks_id)["xyz"]

            next_point = np.roll(points, shift=-1, axis=0)

            feature = np.concatenate([points[:, :2], next_point[:, :2], np.arange(len(next_point))[:, None],
                                      np.ones([len(points), 1]) + idx], axis=1)

            crosswalk_features.extend(feature)

        return np.array(crosswalk_features)

    def build_lane_graph(self, point_dist=3):
        lane_graph = {}

        lane_ids = self.map_api.bounds_info["lanes"]["ids"]

        for lane_id in lane_ids:
            lane_el = self.map_api[lane_id].element.lane

            adj_left = lane_el.adjacent_lane_change_left.id.decode("utf-8")
            adj_right = lane_el.adjacent_lane_change_right.id.decode("utf-8")
            childrens = [c.id.decode("utf-8") for c in lane_el.lanes_ahead]

            lane_dict = self.map_api.get_lane_as_interpolation(lane_id, point_dist, InterpolationMethod.INTER_METER)

            if len(lane_dict['xyz_midlane']) < 2:
                lane_dict = self.map_api.get_lane_as_interpolation(lane_id, 2, InterpolationMethod.INTER_ENSURE_LEN)

            lane_graph[lane_id] = {
                'adj_left': adj_left,
                'adj_right': adj_right,
                'childrens': childrens,
                'mids': lane_dict['xyz_midlane'], #mid is just the mean of left and right
                'lefts': lane_dict['xyz_left'],
                'rights': lane_dict['xyz_right'],
            }

        return lane_graph

    def build_vector_graph(self, lane_graph):
        vector_features = []
        vector_list = []
        vector_id_dict = {}

        for idx, (lane_id, v) in enumerate(lane_graph.items()):

            cur_mids = v['mids']

            # extract left and right mids
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

            # last_dist = []
            # last_next_pos=[]

            last_pos = cur_mids[-1]

            # for child_id in v['childrens']:

            if len(v['childrens']) > 0:

                child_id = v['childrens'][0]

                last_next_pos = lane_graph[child_id]['mids'][0]
            else:
                last_next_pos = last_pos

            last_diff = last_next_pos - last_pos

            child_dist = np.linalg.norm(last_diff, axis=0)

            # for child_id in :
            #     next_first_pos = lane_graph[child_id]['mids'][0]
            #     last_next_pos.append(next_first_pos)#['+hiT', 'diiT', 'fD+t', 'o4kX'] [array([1017.27910042, -261.74356007,  264.00451346]), array([1017.27910042, -261.74356007,  264.00451346]), array([1017.27910042, -261.74356007,  264.00451346])]
            # last_diff=next_first_pos - last_pos
            #
            # child_dist = np.linalg.norm(last_diff, axis=0)
            # last_dist.append(child_dist)

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
                width = min(left_width, right_width)

                dist_to_goal = 0
                tl_feature = 0
                avail = 1

                features = [mid[0], mid[1], next_pos[0], next_pos[1], width, left_dist, right_dist, dist_to_goal,
                            tl_feature, order, avail, lane_id]

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

        return vector_list, vector_id_dict, vector_features

    def build_graphs(self):
        print("build graph")

        crosswalk_features = self.build_crosswalk()

        lane_graph = self.build_lane_graph()  # 8505

        vector_list, vector_id_dict, vector_features = self.build_vector_graph(lane_graph)  # 67004

        vector_features = np.array(vector_features)

        vector_laneids = vector_features[:, -1]  # [:, -1][:, :-1]

        vector_laneids = np.array(vector_laneids)

        vector_features = vector_features[:, :-1].astype(np.float64)

        nei_graph = self.build_neighbor_graph(vector_list, vector_id_dict)

        dist_graph = self.build_dist_graph(nei_graph)  # np.load("./data/lyft_meta/lyft_map_features.npz",allow_pickle=True)['dist_graph']#

        print("build graph, done")

        return dist_graph, vector_features, vector_laneids, crosswalk_features



