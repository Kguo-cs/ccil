import os
import numpy as np
import pathlib
from typing import  cast
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import  PlannerInitialization, PlannerInput
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states,_state_se2_to_ego_state,_get_fixed_timesteps
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

from l5kit.geometry import  transform_points,angular_distance
from l5kit.dataset.utils import move_to_device, move_to_numpy
import torch



class Nuplan_simulator():
    def __init__(self,  model,eval_dataset,model_name):
        self.model = model

        if model_name=="nuplan":
            self.dataset=eval_dataset
            self._future_horizon=eval_dataset.future_horizon
            self._step_interval = eval_dataset.step_time
            self.scenarios_info=eval_dataset.scenarios_info
            self.history_len=eval_dataset.causal_len-1
            self.causal_interval=eval_dataset.causal_interval

            self.vehicle_parameters=VehicleParameters(width=2.297,front_length=4.049,rear_length=1.127,cog_position_from_rear_axle=1.67,height=1.777,wheel_base=3.089,vehicle_name='pacifica',vehicle_type='gen1')

            self.vis = False

            if self.vis:
                self.scenarios=eval_dataset.scenarios
                self.output_directory = pathlib.Path('./vis/nuplan') / 'simulation_log'
        else:
            self._future_horizon = model.future_trajectory_sampling.time_horizon
            self._step_interval = model.future_trajectory_sampling.step_time
            self.feature_builders = model.get_list_of_required_feature()
            self._simulation_history_buffer_duration = 2.0
            self.scenarios = eval_dataset.scenarios

        self.sim_len=250


    def unroll(self,scene_ids,device,model_name):

        sim_outs=[]

        if model_name=="nuplan":
            ego_states=[]

            histories=[]

            info_list=[]

            scenarios=[]

            for scene_idx in scene_ids:

                info=self.scenarios_info[scene_idx]

                if self.vis==True:
                    scenario = self.scenarios[scene_idx]

                    histories.append(SimulationHistory(map_api=scenario.map_api,mission_goal=scenario.get_mission_goal()))

                    scenarios.append(scenario)
                else:
                    histories.append([])

                start_time=int(info[0])

                ego_translations=self.dataset.ego_array[start_time][:2]

                prev_translations=self.dataset.ego_array[start_time-1][:2]

                pp_translations=self.dataset.ego_array[start_time-2][:2]

                timepoint = self.dataset.time_index[start_time]

                prev_timepoint = self.dataset.time_index[start_time-1]

                pp_timepoint = self.dataset.time_index[start_time-2]

                cur_time_gap=(timepoint-prev_timepoint)/1e6

                prev_time_gap=(prev_timepoint-pp_timepoint)/1e6

                world_vel=(ego_translations-prev_translations)/cur_time_gap

                prev_vel=(prev_translations-pp_translations)/prev_time_gap

                world_acc=(world_vel-prev_vel)/cur_time_gap

                ego_yaws=self.dataset.ego_array[start_time][2]

                prev_yaws=self.dataset.ego_array[start_time-1][2]

                pp_yaws=self.dataset.ego_array[start_time-2][2]

                angular_vel=angular_distance(ego_yaws,prev_yaws)/cur_time_gap

                prev_angular_vel=angular_distance(prev_yaws,pp_yaws)/prev_time_gap

                angular_accel=(angular_vel-prev_angular_vel)/cur_time_gap

                initial_ego_state=EgoState.build_from_rear_axle(
                                rear_axle_pose=StateSE2(ego_translations[0], ego_translations[1], ego_yaws),
                                rear_axle_velocity_2d=StateVector2D(world_vel[0], world_vel[1]),
                                rear_axle_acceleration_2d=StateVector2D(world_acc[0], world_acc[1]),
                                tire_steering_angle=0.0,
                                time_point=TimePoint(timepoint),
                                vehicle_parameters=self.vehicle_parameters,
                                angular_vel=angular_vel,
                                angular_accel=angular_accel,

                )

                ego_states.append(initial_ego_state)

                info_list.append(info)

            state_embeddings_list=[]

            for inter in range(self.causal_interval,0,-1):

                ego_input = []

                for info in info_list:
                    lane_polylines = []
                    crosswalk_polylines = []
                    agent_polylines = []
                    ego_polyline = []

                    for iteration_index in range(-self.history_len*inter,-self.history_len*(inter-1)):
                        lane_polylines_t,crosswalk_polylines_t,agent_polylines_t,ego_polyline_t,future_coords_offset,future_yaws_offset = self.dataset.get_frame(iteration_index, info)

                        lane_polylines.append(lane_polylines_t)
                        crosswalk_polylines.append(crosswalk_polylines_t)
                        agent_polylines.append(agent_polylines_t)
                        ego_polyline.append(ego_polyline_t)

                    data = {
                        "lane_polylines": np.array(lane_polylines),
                        "crosswalk_polylines": np.array(crosswalk_polylines),
                        "agent_polylines": np.array(agent_polylines),
                        "ego_polyline": np.array(ego_polyline)
                    }

                    ego_input.append(data)

                ego_input_dict = default_collate(ego_input)

                ego_input_dict_device = move_to_device(ego_input_dict, device)

                state_embeddings=self.model.embed_state(ego_input_dict_device)[0]

                state_embeddings_list.append(state_embeddings)

            state_embeddings=torch.cat(state_embeddings_list,dim=1)

            for iteration_index in tqdm(range(self.sim_len), disable=False):

                ego_input = []

                tracked_objects_all=[]

                for info,ego_state in zip(info_list,ego_states):

                    data,tracked_objects = self.dataset.get_frame(iteration_index,info,ego_state)

                    ego_input.append(data)

                    tracked_objects_all.append(tracked_objects)

                ego_input_dict = default_collate(ego_input)

                ego_input_dict_device = move_to_device(ego_input_dict, device)

                ego_output_dict, state_embeddings = self.model.get_action(ego_input_dict_device, state_embeddings)

                ego_output_dict=move_to_numpy(ego_output_dict)

                world_from_agent = ego_input_dict["world_from_agent"].numpy()

                ego_translations = transform_points(ego_output_dict["positions"], world_from_agent)

                ego_yaws = ego_input_dict["yaw"].numpy()[:,None,None] + ego_output_dict["yaws"]

                for i,ego_state in enumerate(ego_states):

                    if self.vis:

                        timesteps = _get_fixed_timesteps(ego_state, self._future_horizon, self._step_interval)

                        predicted_poses = np.concatenate([ego_translations[i], ego_yaws[i]], axis=-1)

                        absolute_states = [StateSE2.deserialize(pose) for pose in predicted_poses]

                        states = [
                            _state_se2_to_ego_state(state, timestep, ego_state.car_footprint.vehicle_parameters)
                            for state, timestep in zip(absolute_states, timesteps)
                        ]

                        states.insert(0, ego_state)

                        trajectory = InterpolatedTrajectory(states)

                        scenario=scenarios[i]

                        traffic_light_status = list(scenario.get_traffic_light_status_at_iteration(iteration_index))

                        observation = scenario.get_tracked_objects_at_iteration(iteration_index)

                        iteration = SimulationIteration(time_point=scenario.get_time_point(iteration_index), index=iteration_index)

                        histories[i].add_sample(
                            SimulationHistorySample(iteration, ego_state, trajectory, observation,
                                                    traffic_light_status))
                    else:
                        observation = tracked_objects_all[i]

                        histories[i].append((ego_state,observation))

                    if iteration_index!=self.sim_len-1:

                        next_time=self.dataset.time_index[int(info_list[i][0])+iteration_index+1]

                        real_time=next_time-ego_state.time_us

                        pred_time = np.arange(1, ego_translations.shape[1] + 1) * 1e5

                        x = np.interp(real_time, pred_time, ego_translations[i,:, 0])
                        y = np.interp(real_time, pred_time, ego_translations[i,:, 1])

                        cur_time_gap=real_time/(1e6)

                        world_vel=(np.array([x,y])-ego_state.rear_axle.array)/cur_time_gap

                        world_acc = (world_vel - ego_state.dynamic_car_state.rear_axle_velocity_2d.array) / cur_time_gap

                        yaw=np.interp(real_time, pred_time, ego_yaws[i,:,0])

                        heading=angular_distance(yaw,0)

                        angular_vel=angular_distance(ego_yaws[i][0][0],ego_state.rear_axle.heading)/cur_time_gap

                        angular_accel = (angular_vel - ego_state.dynamic_car_state.angular_velocity) / cur_time_gap

                        ego_states[i]=EgoState.build_from_rear_axle(
                                rear_axle_pose=StateSE2(x, y, heading),
                                rear_axle_velocity_2d=StateVector2D(world_vel[0], world_vel[1]),
                                rear_axle_acceleration_2d=StateVector2D(world_acc[0], world_acc[1]),
                                tire_steering_angle=0.0,
                                time_point=TimePoint(next_time),
                                vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                                angular_vel=angular_vel,
                                angular_accel=angular_accel
                        )
                    else:

                        start_time=int(info_list[i][0])

                        expert_traj=self.dataset.ego_array[start_time:start_time+600,:2]#

                        sim_outs.append((histories[i], torch.tensor(expert_traj),scene_ids[i]))

                        if self.vis==True:
                            scenario = scenarios[i]

                            planner_name = "Transformer"

                            file_suffix = '.msgpack.xz'

                            scenario_directory = self.output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name

                            if not os.path.exists(scenario_directory):
                                os.makedirs(scenario_directory)

                            file_name = scenario_directory / str(scene_ids[i])#scenario.scenario_name

                            file_name = file_name.with_suffix(file_suffix)

                            simulation_log = SimulationLog(file_path=file_name, scenario=scenario, planner=self.model,
                                                           simulation_history=histories[i])

                            simulation_log.save_to_file()

        else:

            for scene_idx in scene_ids:
                sim_outs.append(self.nuplan_unroll(scene_idx,device))

        return sim_outs

    def nuplan_unroll(self,scene_idx,device):

        scenario = self.scenarios[scene_idx]

        _initialization = PlannerInitialization(
            expert_goal_state=scenario.get_expert_goal_state(),
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        # History where the steps of a simulation are stored
        history = []  # SimulationHistory(scenario.map_api, scenario.get_mission_goal())

        _history_buffer_size = int(self._simulation_history_buffer_duration // scenario.database_interval + 1)

        _history_buffer = SimulationHistoryBuffer.initialize_from_scenario(_history_buffer_size, scenario,
                                                                           DetectionsTracks)

        ego_state = scenario.initial_ego_state

        for iteration_index in tqdm(range(self.sim_len), disable=False):

            observation = scenario.get_tracked_objects_at_iteration(iteration_index)

            _history_buffer.append(ego_state, observation)

            traffic_light_status = scenario.get_traffic_light_status_at_iteration(iteration_index)

            iteration = SimulationIteration(time_point=scenario.get_time_point(iteration_index),
                                            index=iteration_index)

            planner_input = PlannerInput(iteration=iteration, history=_history_buffer,
                                         traffic_light_data=traffic_light_status)

            features = {
                builder.get_feature_unique_name(): builder.get_features_from_simulation(planner_input,
                                                                                        _initialization)
                for builder in self.feature_builders
            }

            features = {name: feature.to_feature_tensor() for name, feature in features.items()}
            features = {name: feature.to_device(device) for name, feature in features.items()}
            features = {name: feature.collate([feature]) for name, feature in features.items()}

            predictions = self.model.forward(features)

            # Extract trajectory prediction
            trajectory_predicted = cast(Trajectory, predictions['trajectory'])
            trajectory_tensor = trajectory_predicted.data
            predictions = trajectory_tensor.cpu().detach().numpy()[0].astype(
                np.float32)  # retrive first (and only) batch as a numpy array

            states = transform_predictions_to_states(predictions, ego_state, self._future_horizon,
                                                     self._step_interval)
            trajectory = InterpolatedTrajectory(states)

            tracked_objects = np.zeros([len(observation.tracked_objects.tracked_objects), 6])

            for i, agent in enumerate(observation.tracked_objects.tracked_objects):
                tracked_objects[i][:2] = agent.center.array
                tracked_objects[i][2] = agent.center.heading
                tracked_objects[i][3] = agent.box.width
                tracked_objects[i][4] = agent.box.length
                tracked_objects[i][5] = agent.box.height

            history.append((ego_state,
                            tracked_objects))  # .add_sample(SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status))

            if iteration_index != self.sim_len - 1:
                ego_state = trajectory.get_state_at_time(scenario.get_time_point(iteration_index + 1))

        expert_traj = scenario.get_expert_ego_trajectory()

        expert_state = torch.stack([torch.tensor(state.rear_axle.array) for state in expert_traj],
                                   dim=0)

        return (history, expert_state ,scene_idx)

