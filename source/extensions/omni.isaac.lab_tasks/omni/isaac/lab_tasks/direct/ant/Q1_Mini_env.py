# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.math import sample_uniform

# from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from omni.isaac.lab.actuators import ImplicitActuatorCfg

gear = 1

Q1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Suranjan/AppData/Local/ov/pkg/isaac-lab/IsaacLab/Q1_Mini/Q1_Assembly/Q1_Mini_Test.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Position slightly above ground for initialization
        # joint_pos={
        #     "BASE_Revolute_1": 0.0,
        #     "BASE_Revolute_3": 0.0,
        #     "BASE_Revolute_5": 0.0,
        #     "BASE_Revolute_7": 0.0,
        #     "Coxa_BL_Revolute_21": 0.0,
        #     "Coxa_FR_Revolute_4": 0.0,
        #     "Coxa_FL_Revolute_6": 0.0,
        #     "Coxa_BR_Revolute_8": 0.0,
        # },

    ),
    actuators={
        "base_actuator_1": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_1"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=5.0,
        ),
        "base_actuator_3": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_3"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "base_actuator_5": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_5"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "base_actuator_7": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_7"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "coxa_actuator_BL": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_BL_Revolute_21"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "coxa_actuator_FR": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_FR_Revolute_4"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "coxa_actuator_FL": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_FL_Revolute_6"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
        "coxa_actuator_BR": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_BR_Revolute_8"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=0.0,
            damping=50.0,
        ),
    },
)


## Configuration for the Q1 Mini Robot ##
@configclass
class Q1MiniEnvCfg(DirectRLEnvCfg):
    # Env parameters
    episode_length_s = 10.0
    decimation = 2
    action_scale = 1
    action_space = 8            # 8 servo position references
    observation_space = 8       # 8 previous references
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = Q1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    dof_names =  [  "BASE_Revolute_1",
                    "BASE_Revolute_3",
                    "BASE_Revolute_5",
                    "BASE_Revolute_7",
                    "Coxa_BL_Revolute_21",
                    "Coxa_FR_Revolute_4",
                    "Coxa_FL_Revolute_6",
                    "Coxa_BR_Revolute_8"]

    rew_scale_velocity = 1.0
    rew_scale_heading = 1.0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=.50, replicate_physics=True)




class Q1MiniEnv(DirectRLEnv):
    cfg: Q1MiniEnvCfg

    def __init__(self, cfg: Q1MiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.initialize_dof_indices()
        self.action_scale = self.cfg.action_scale
        self.prev_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def initialize_dof_indices(self):
        # Attempt to find joint indices by names, ensure all are found
        joint_indices = []
        missing_joints = []
        for joint_name in self.cfg.dof_names:
            index, found = self.robot.find_joints(joint_name)
            if found:
                joint_indices.append(index)
            else:
                missing_joints.append(joint_name)

        if missing_joints:
            raise ValueError(f"Missing joints in robot configuration: {missing_joints}")

        # Convert list of indices to a tensor
        self._dof_idx = torch.tensor(joint_indices, dtype=torch.long, device=self.robot.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Now 'self.robot' should be correctly initialized and available
        self.actions = self.cfg.action_scale * actions  # Scale actions appropriately
        # Increment the step counter
        # self.actions = actions.clone()

    def _apply_action(self):
        # Assuming self.actions is [2048, 8] and self._dof_idx is [8]
        # You may need to ensure `actions` is expanded or reshaped to fit the expected input for set_joint_effort_target
        # This reshaping depends on how the Articulation method expects the data

        # Example: If the method expects a specific shaping of the tensor:
        actions_to_apply = self.actions.unsqueeze(2)  # This would adjust shape to [2048, 8, 1]

        # Then use this reshaped actions to apply:
        self.robot.set_joint_effort_target(actions_to_apply, joint_ids=self._dof_idx)

    def _get_observations(self) -> dict:
        # Observation is the previously applied actions
        observations = self.prev_action
        self.prev_action = self.actions  # Update previous actions for next step
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        # velocity = self.robot.data.root_lin_vel_w  # Assuming velocity along x-direction is desired

        # Simple reward function that rewards forward movement and correct heading
        progress_reward = self.potentials - self.prev_potentials

        # to_target = self.targets - self.robot.data.root_pos_w
        # to_target[:, 2] = 0.0
        # _, _, heading_proj, _, _ = compute_heading_and_up(
        #     self.robot.data.root_quat_w, quat_conjugate(self.start_rotation).repeat((self.num_envs, 1)), to_target, self.basis_vec0, self.basis_vec1, 2
        # )
        # heading_weight_tensor = torch.ones_like(heading_proj) * self.cfg.rew_scale_heading
        # reward_heading = torch.where(heading_proj > 0.8, heading_weight_tensor, self.cfg.rew_scale_heading * heading_proj / 0.8)
        return progress_reward
        #+ reward_heading

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for joint limits
        to_target = self.targets - self.robot.data.root_pos_w
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        out_of_bounds_lower = self.robot.data.joint_pos*0.98 < lower_limits
        out_of_bounds_upper = self.robot.data.joint_pos*0.98 > upper_limits
        out_of_bounds = torch.any(torch.logical_or(out_of_bounds_lower, out_of_bounds_upper), dim=1)

        # Check if the episode time has exceeded the specified length
        # This assumes that your simulation steps in the environment are counted in `_pre_physics_step` or tracked elsewhere
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt
        self.prev_potentials[:] = self.potentials



