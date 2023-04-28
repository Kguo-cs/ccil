from l5kit.simulation.unroll import ClosedLoopSimulatorModes, SimulationOutput, UnrollInputOutput
from collections import defaultdict
from typing import Dict,  List, Optional,  Set, Tuple
from l5kit.geometry.transform import yaw_as_rotation33

import numpy as np
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
import torch

class ClosedLoopSimulator:
    def __init__(self, sim_cfg: SimulationConfig,
                 dataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None,
                 keys_to_exclude: Tuple[str] = ("image",),
                 mode: int = ClosedLoopSimulatorModes.L5KIT):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
        :param mode: the framework that uses the closed loop simulator
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("agents model should not be None when simulating agent")
        if sim_cfg.use_ego_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("ego has to be simulated when using gym environment")
        if not sim_cfg.use_agents_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("agents need be log-replayed when using gym environment")

        self.model_ego = torch.nn.Sequential().to(device) if model_ego is None else model_ego.to(device)
        self.model_agents = torch.nn.Sequential().to(device) if model_agents is None else model_agents.to(device)

       # self.device = device
        self.dataset = dataset

        self.keys_to_exclude = {"lane_polylines", 'crosswalk_polylines',  'agent_polylines', "ego_polyline"}

        self.start_frame_index=1

        self.sim_len=249

    def unroll(self, scene_indices: List[int],device, model_name) -> List[SimulationOutput]:
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self.sim_cfg)

        agents_ins_outs = defaultdict(list)
        ego_ins_outs = defaultdict(list)

        for frame_index in tqdm(range(self.sim_len), disable=False):  # range(len(sim_dataset))

            ego_input = []
            for scene_idx, scene_dt in sim_dataset.scene_dataset_batch.items():

                state_index=min(frame_index, len(scene_dt.dataset.frames)-1)

                if  model_name=="lyft":
                    data = self.dataset.get_frame(scene_idx, state_index, scene_dt.dataset)
                else:
                    data = scene_dt[state_index]

                data["scene_index"] = scene_idx  # set the scene to the right index

                ego_input.append(data)

            ego_input_dict = default_collate(ego_input)

            ego_input_dict_device = move_to_device(ego_input_dict, device)

            if frame_index < self.start_frame_index:
                ego_output_dict = {}
                ego_output_dict["positions"] = ego_input_dict_device["target_positions"]
                ego_output_dict["yaws"] = ego_input_dict_device["target_yaws"]

                if model_name == "lyft":
                    state_embeddings=self.model_ego.embed_state(ego_input_dict_device)[0]
            else:
                if model_name == "lyft":
                    ego_output_dict, state_embeddings = self.model_ego.get_action(ego_input_dict_device, state_embeddings)
                else:
                    ego_output_dict = self.model_ego(move_to_device(ego_input_dict_device, device))

            ego_output_dict = move_to_numpy(ego_output_dict)
            ego_input_dict = move_to_numpy(ego_input_dict)

            next_frame_index = frame_index + 1

            should_update = next_frame_index != self.sim_len

            if should_update:
                self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

            ego_frame_in_out = self.get_ego_in_out(ego_input_dict, ego_output_dict, self.keys_to_exclude)

            for scene_idx in scene_indices:
                ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_dataset, ego_ins_outs, agents_ins_outs))

        return simulated_outputs

    @staticmethod
    def get_ego_in_out(input_dict: Dict[str, np.ndarray],
                       output_dict: Dict[str, np.ndarray],
                       keys_to_exclude: Optional[Set[str]] = None) -> Dict[int, UnrollInputOutput]:
        """Get all ego inputs and outputs as a dict mapping scene index to a single UnrollInputOutput

        :param input_dict: all ego model inputs (across scenes)
        :param output_dict: all ego model outputs (across scenes)
        :param keys_to_exclude: if to drop keys from input/output (e.g. huge blobs)
        :return: the dict mapping scene index to a single UnrollInputOutput.
        """
        key_required = {"track_id", "scene_index"}
        if len(key_required.intersection(input_dict.keys())) != len(key_required):
            raise ValueError(f"track_id and scene_index not found in keys {input_dict.keys()}")

        keys_to_exclude = keys_to_exclude if keys_to_exclude is not None else set()
        if len(key_required.intersection(keys_to_exclude)) != 0:
            raise ValueError(f"can't drop required keys: {keys_to_exclude}")

        ret_dict = {}
        scene_indices = input_dict["scene_index"]
        if len(np.unique(scene_indices)) != len(scene_indices):
            raise ValueError(f"repeated scene_index for ego! {scene_indices}")

        for idx_ego in range(len(scene_indices)):
            ego_in = {k: v[idx_ego] for k, v in input_dict.items() if k not in keys_to_exclude}
            ego_out = {k: v[idx_ego] for k, v in output_dict.items() if k not in keys_to_exclude}
            ret_dict[ego_in["scene_index"]] = UnrollInputOutput(track_id=ego_in["track_id"],
                                                                inputs=ego_in,
                                                                outputs=ego_out)
        return ret_dict

    @staticmethod
    def update_ego(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                   output_dict: Dict[str, np.ndarray]) -> None:
        """Update ego across scenes for the given frame index.

        :param dataset: The simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the ego model
        :param output_dict: the output of the ego model
        :return:
        """
        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        ego_translations = transform_points(output_dict["positions"][:, :], world_from_agent)
        ego_yaws = np.expand_dims(yaw, -1) + output_dict["yaws"][:, :, 0]

        if "next_timestep" in input_dict.keys():
            next_timestep = input_dict["next_timestep"]
        else:
            next_timestep=np.zeros([len(ego_translations)])+1e8

        pred_time = np.arange(1, ego_translations.shape[1] + 1) * 1e8

        for i, (scene_dataset, ego_translation, ego_yaw,real_time) in enumerate(
            zip(dataset.scene_dataset_batch.values(), ego_translations, ego_yaws,next_timestep)
        ):
            if frame_idx<len(scene_dataset.dataset.frames):
                x = np.interp(real_time, pred_time, ego_translation[:, 0])
                y = np.interp(real_time, pred_time, ego_translation[ :, 1])

                angle_rad = np.interp(real_time, pred_time, ego_yaw)

                scene_dataset.dataset.frames[frame_idx]["ego_translation"][0] = x
                scene_dataset.dataset.frames[frame_idx]["ego_translation"][1] = y

                scene_dataset.dataset.frames[frame_idx]["ego_rotation"] = yaw_as_rotation33(angle_rad)

