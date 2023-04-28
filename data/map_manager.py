from tqdm import tqdm
import numpy as np
import heapq

class MapManager:
    def cur_lane_dist(self,tree,dist_graph, cur_pos, lane_indices,cur_query_dist=7):

        cur_indices = tree.query_ball_point(cur_pos, cur_query_dist)  # k-nearest element

        if len(cur_indices) == 0:
            cur_dists, cur_indices = tree.query(cur_pos, 2)

        cur_indices = np.array(cur_indices)
        lane_indices = np.array(lane_indices)

        cur2lane = dist_graph[cur_indices[:, None], lane_indices[None]]

        lane2cur = dist_graph[lane_indices[:, None], cur_indices[None]]

        return np.minimum(cur2lane, lane2cur.T).min(0)


    def connect_nearest(self,cur_mids, adj_mids):  # find closest mids in left and right if exists
        rel_pos = cur_mids[:, None] - adj_mids[None]
        dist = np.linalg.norm(rel_pos, axis=2)

        connect_ids = np.argmin(dist, axis=1)

        connect_dists = np.min(dist, axis=1)

        return connect_ids, connect_dists

    def build_neighbor_graph(self,vector_list, vector_id_dict):

        nei_graph = []

        for vector_node in vector_list:

            neighbors = []

            if vector_node['left_dist'] != 0:
                left_id = vector_node['left_id']

                left_node = vector_id_dict[left_id]
                left_dist = vector_node['left_dist']
                neighbors.append((left_node, left_dist))

            if vector_node['right_dist'] != 0:
                right_id = vector_node['right_id']

                right_node = vector_id_dict[right_id]
                right_dist = vector_node['right_dist']
                neighbors.append((right_node, right_dist))

            next_dist = vector_node['next_dist']

            for next_id in vector_node["next_id"]:
                next_node = vector_id_dict[next_id]

                neighbors.append((next_node, next_dist))

            nei_graph.append(neighbors)

        return nei_graph

    def build_dist_graph(self,graph, max_dist=60000):

        graph_len = len(graph)

        dist_graph = np.zeros([graph_len, graph_len], dtype=np.uint16)

        for start_node in tqdm(range(graph_len)):

            dist = np.zeros([graph_len]) + max_dist

            dist[start_node] = 0

            visited = [False] * graph_len

            pq = [(0, start_node)]

            while len(pq) > 0:

                min_dist, u = heapq.heappop(pq)

                if min_dist > max_dist:
                    break

                if visited[u]:
                    continue

                visited[u] = True

                for v, l in graph[u]:

                    if dist[u] + l < dist[v]:
                        dist[v] = dist[u] + l
                        heapq.heappush(pq, (dist[v], v))

           # print(start_node, np.max(dist))

            dist_graph[start_node] = np.round(dist).astype('uint16')

        return dist_graph

    def min_dist_to_goal(self,vector_features,tree,cur_pos,dist_graph,cur_query_dist=7):

        cur_indices = tree.query_ball_point(cur_pos, cur_query_dist)  # k-nearest element

        if len(cur_indices) == 0:
            cur_dists, cur_indices = tree.query(cur_pos, 2)

        vector = vector_features[cur_indices, :2].astype(np.float32)

        cur_dist_to_vector = np.linalg.norm(vector - cur_pos[None], axis=1)

        cur_to_goal = dist_graph[cur_indices] + cur_dist_to_vector[:, None]

        min_dist = np.min(cur_to_goal, axis=0)

        return min_dist

