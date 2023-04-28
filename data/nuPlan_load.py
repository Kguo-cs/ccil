from typing import Optional
import numpy as np
from pyquaternion import Quaternion
from nuplan.common.actor_state.tracked_objects_types import  TrackedObjectType
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping
from l5kit.configs import load_config_data
import sqlite3
import os
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import (
    discover_log_dbs,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioMapping,absolute_path_to_log_name
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario,ScenarioExtractionInfo
from multiprocessing import Pool



split_cfg = load_config_data("./data/nuplan_meta/nuplan.yaml")

map_root=os.environ["NUPLAN_DATA_FOLDER"]+"/maps"
map_version = 'nuplan-maps-v1.0'

vehicle_parameters=VehicleParameters(width=2.297,front_length=4.049,rear_length=1.127,cog_position_from_rear_axle=1.67,height=1.777,wheel_base=3.089,vehicle_name='pacifica',vehicle_type='gen1')

sim_len=250
scenario_duration=sim_len/10
extraction_offset=-7.0
history_len=3

def collate(file):
    scenario_list=[]

    ego_list = []

    agent_list= []

    green_tl_list = []

    red_tl_list=[]

    source = sqlite3.connect( file)
    dest = sqlite3.connect(':memory:')
    source.backup(dest)

    dest.row_factory = sqlite3.Row
    cursor = dest.cursor()

    filter_clause = ""

    query = f"""
        WITH ordered_scenes AS
        (
            SELECT  token,
                    ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT  COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT  o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n

            -- Define "valid" scenes as those that have at least 2 before and 2 after
            -- Note that the token denotes the beginning of a scene
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT  lp.token,
                lp.timestamp,
                l.map_version AS map_name,

                -- scenarios can have multiple tags
                -- Pick one arbitrarily from the list of acceptable tags
                MAX(st.type) AS scenario_type
        FROM lidar_pc AS lp
        LEFT OUTER JOIN scenario_tag AS st
            ON lp.token = st.lidar_pc_token
        INNER JOIN lidar AS ld
            ON ld.token = lp.lidar_token
        INNER JOIN log AS l
            ON ld.log_token = l.token
        INNER JOIN valid_scenes AS vs
            ON lp.scene_token = vs.token
        {filter_clause}
        GROUP BY    lp.token,
                    lp.timestamp,
                    l.map_version
        ORDER BY lp.timestamp ASC;
    """

    cursor.execute(query, [])

    time_steps = []

    log_name=absolute_path_to_log_name(file)

    data_root = os.path.dirname(file)

    for iteration,row in enumerate(cursor):

        if row["map_name"]!="us-nv-las-vegas-strip":
            break

        if iteration % 2 != 0:
            continue

        time_steps.append(row["timestamp"])
        scenario_type = row["scenario_type"]

        if scenario_type is None:
            scenario_type = DEFAULT_SCENARIO_NAME


        extraction_info = ScenarioExtractionInfo(
                scenario_name=scenario_type,
                scenario_duration=scenario_duration,
                extraction_offset=extraction_offset,
                subsample_ratio=0.5,
            )

        scenario=NuPlanScenario(
            data_root,
            log_name,
            row["token"].hex(),
            row["timestamp"],
            scenario_type,
            map_root,
            map_version,
            row["map_name"],
            extraction_info,  # if expand_scenerios is none, only first scenario will be considered
            vehicle_parameters,
            None,
        )

        scenario_list.append(scenario)

    if len(scenario_list) > 0:
        start_timestamp = time_steps[0] + (extraction_offset-history_len-0.01) * 1e6     #past 5 s

        query = """
        WITH numbered AS
        (
            SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
            FROM lidar_pc
            WHERE timestamp >= ?
            AND timestamp <= ?
        )
        SELECT token
        FROM numbered
        WHERE ((row_num - 1) % ?) = 0
        ORDER BY timestamp ASC;
        """

        end_timestamp = time_steps[-1] + 60 * 1e6  #future 60 s

        subsample_interval = 1

        query_parameters = (start_timestamp, end_timestamp, subsample_interval)

        cursor.execute(query, query_parameters)

        _lidarpc_tokens = []

        for row in cursor:
            lidarpc_tokens = row["token"].hex()
            _lidarpc_tokens.append(lidarpc_tokens)

        # query_parameters = (bytearray.fromhex(_lidarpc_tokens[0]),)
        #
        # query = """
        # SELECT timestamp
        # FROM lidar_pc
        # WHERE token = ?
        # """
        #
        # cursor.execute(query, query_parameters)
        #
        # row: Optional[sqlite3.Row] = cursor.fetchone()
        #
        # time_step = row["timestamp"]
        #
        # initial_iteration=round((time_step-time_steps[0])*2e-5)
        #
        # query_parameters = (bytearray.fromhex(_lidarpc_tokens[-1]),)
        #
        # query = """
        # SELECT timestamp
        # FROM lidar_pc
        # WHERE token = ?
        # """
        #
        # cursor.execute(query, query_parameters)
        #
        # row: Optional[sqlite3.Row] = cursor.fetchone()
        #
        # time_step = row["timestamp"]
        #
        # end_iteration=round((time_step-time_steps[-1])*2e-5)
        #
        # print(initial_iteration,end_iteration)

        # if end_iteration<400:
        #     return ego_list, agent_list, tl_list, route_list, [], []

        for iteration,token in enumerate(_lidarpc_tokens):#time_steps,times_step,
            if iteration%2!=0:
                continue

            query_parameters = (bytearray.fromhex(token),)

            query = """
                SELECT  c.name AS category_name,
                        lb.x,
                        lb.y,
                        lb.z,
                        lb.yaw,
                        lb.width,
                        lb.length,
                        lb.height,
                        lb.vx,
                        lb.vy,
                        lb.token,
                        lb.track_token,
                        lp.timestamp
                FROM lidar_box AS lb
                INNER JOIN track AS t
                    ON t.token = lb.track_token
                INNER JOIN category AS c
                    ON c.token = t.category_token
                INNER JOIN lidar_pc AS lp
                    ON lp.token = lb.lidar_pc_token
                WHERE lp.token = ?
            """

            cursor.execute(query, query_parameters)

            for row in cursor:
                category_name = row["category_name"]

                label_local = raw_mapping["global2local"][category_name]
                tracked_object_type = TrackedObjectType[local2agent_type[label_local]]

                feature = [row["x"], row["y"], row["yaw"],
                           row["width"], row["length"], row["height"],
                           row["vx"], row["vy"],
                           tracked_object_type.value,
                           hash(row["track_token"])]

                agent_list.append(feature)

            ego_states = np.zeros([9])

            query = """
                SELECT  ep.x,
                        ep.y,
                        ep.qw,
                        ep.qx,
                        ep.qy,
                        ep.qz,
                        -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!
                        -- use lidar_pc timestamp for backwards compatibility.
                        lp.timestamp,
                        ep.vx,
                        ep.vy,
                        ep.acceleration_x,
                        ep.acceleration_y
                FROM ego_pose AS ep
                INNER JOIN lidar_pc AS lp
                    ON lp.ego_pose_token = ep.token
                WHERE lp.token = ?
            """

            cursor.execute(query, query_parameters)

            row: Optional[sqlite3.Row] = cursor.fetchone()

            q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])

            ego_states[0] = row["x"]#rear axis not center
            ego_states[1] = row["y"]#
            ego_states[2] = q.yaw_pitch_roll[0]

            ego_states[3]=len(agent_list)

            query = """
                SELECT  CASE WHEN tl.status == "green" THEN 0
                             WHEN tl.status == "yellow" THEN 1
                             WHEN tl.status == "red" THEN 2
                             ELSE 3
                        END AS status,
                        tl.lane_connector_id,
                        lp.timestamp AS timestamp
                FROM lidar_pc AS lp
                INNER JOIN traffic_light_status AS tl
                    ON lp.token = tl.lidar_pc_token
                WHERE lp.token = ?
            """
            cursor.execute(query, query_parameters)

            for row in cursor:
                status = row["status"]
                lane_connector_id = row["lane_connector_id"]

                if status == 2:
                    red_tl_list.append(lane_connector_id)
                elif status == 0:
                    green_tl_list.append(lane_connector_id)

            ego_states[4]=len(red_tl_list)

            ego_states[5]=len(green_tl_list)

            query = """
            SELECT timestamp
            FROM lidar_pc
            WHERE token = ?
            """

            cursor.execute(query, query_parameters)

            row: Optional[sqlite3.Row] = cursor.fetchone()

            time_step = row["timestamp"]

            ego_states[6] = time_step

            query = """
                SELECT  ep.x,
                        ep.y,
                        ep.qw,
                        ep.qx,
                        ep.qy,
                        ep.qz
                FROM ego_pose AS ep
                INNER JOIN scene AS s
                    ON s.goal_ego_pose_token = ep.token
                INNER JOIN lidar_pc AS lp
                    ON lp.scene_token = s.token
                WHERE lp.token = ?
            """

            cursor.execute(query, query_parameters)

            row: Optional[sqlite3.Row] = cursor.fetchone()

            if row is not None:

                ego_states[7] = row["x"]
                ego_states[8] = row["y"]
                # q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
                 # ego_states[9] = q.yaw_pitch_roll[0]

            ego_list.append(ego_states)

    scenario_info=[]
    goal_scenario_list=[]

    for iteration,scenario in enumerate(scenario_list):

        start_time=iteration+history_len*10

        goal_translation=ego_list[start_time+sim_len-1][-2:]

        if goal_translation[0]!=0:

            goal_scenario_list.append(scenario)
            scenario_info.append([start_time,goal_translation[0],goal_translation[1]])

    scenario_info=np.array(scenario_info)
    agent_list=np.array(agent_list)
    red_tl_list=np.array(red_tl_list).astype(np.int32)
    green_tl_list=np.array(green_tl_list).astype(np.int32)
    ego_list=np.array(ego_list)

    return ego_list,agent_list,red_tl_list,green_tl_list,goal_scenario_list,scenario_info


def db_process(type,data_root):

    _db_files=discover_log_dbs(data_root)

    processed_file=[]

    for file in _db_files:
        log_name = absolute_path_to_log_name(file)

        if log_name in split_cfg['log_splits'][type]:
            processed_file.append(file)

    if len(processed_file) == 0:
        return

    # if type=="train":
    #     processed_file=sample(_db_files, 2000)

    print(processed_file,len(processed_file))

    with Pool(len(os.sched_getaffinity(0))) as pool:
        results = pool.map(collate, processed_file)


    ego_array = []

    agent_array = []

    red_tl_array = []

    green_tl_array=[]

    scenarios = []

    scenarios_info = []

    for result in results:
        ego_list,agent_list,red_tl_list,green_tl_list,goal_scenario_list,scenario_info = result

        if len(goal_scenario_list) > 0:
            scenario_info[:,0]+=len(ego_array)
            ego_list[:,3]+=len(agent_array)
            ego_list[:,4]+=len(red_tl_array)
            ego_list[:,5]+=len(green_tl_array)

            ego_array.extend(ego_list)
            agent_array.extend(agent_list)
            red_tl_array.extend(red_tl_list)
            green_tl_array.extend(green_tl_list)

            scenarios.extend(goal_scenario_list)#)[::sim_len]
            scenarios_info.extend(scenario_info)#)[::sim_len]

    ego_array = np.array(ego_array)
    agent_array = np.array(agent_array)
    red_tl_array = np.array(red_tl_array)
    green_tl_array= np.array(green_tl_array)
    scenarios_info=np.array(scenarios_info)

    print(type, "dump", len(scenarios))

    return ego_array,agent_array,red_tl_array,green_tl_array,scenarios,scenarios_info