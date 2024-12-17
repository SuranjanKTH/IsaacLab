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


servo_effort_limit = 1.0
servo_velocity_limit = 0.001
servo_damping = 1.0
servo_stiffness = 10.0
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
    ),
    actuators={
        "base_actuator_1": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_1"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "base_actuator_3": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_3"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "base_actuator_5": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_5"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "base_actuator_7": ImplicitActuatorCfg(
            joint_names_expr=["BASE_Revolute_7"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "coxa_actuator_BL": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_BL_Revolute_21"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "coxa_actuator_FR": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_FR_Revolute_4"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "coxa_actuator_FL": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_FL_Revolute_6"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
        ),
        "coxa_actuator_BR": ImplicitActuatorCfg(
            joint_names_expr=["Coxa_BR_Revolute_8"],
            effort_limit=servo_effort_limit,
            velocity_limit=servo_velocity_limit,
            stiffness=servo_stiffness,
            damping=servo_damping,
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
    observation_space = 24       # 16 previous joints positions + quaternion orientation
    state_space = 0
    time_steps = episode_length_s*60

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120
                                       , render_interval=decimation
    )

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

    rew_scale_progress = 1.0
    rew_scale_heading = 0.5
    rew_scale_upright = 0.10
    rew_scale_energy = 0.005
    pen_scale_symmetry = 0.0001
    pen_scale_unrealistic = 0.0001
    pen_scale_neutral = 0.050           #Penalty for deviations from servo neutral positions
    termination_height=0.032


    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=.50, replicate_physics=True)




class Q1MiniEnv(DirectRLEnv):
    cfg: Q1MiniEnvCfg

    def __init__(self, cfg: Q1MiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.initialize_dof_indices()
        self.action_scale = self.cfg.action_scale
        self.prev_joint_positions = torch.zeros((self.num_envs, len(self.cfg.dof_names), 2), device=self.sim.device)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        # Randomize heading vectors in the xy-plane
        angles = torch.rand(self.num_envs, device=self.sim.device) * 2 * torch.pi
        self.heading_vec = torch.stack(
            (torch.cos(angles), torch.sin(angles), torch.zeros(self.num_envs, device=self.sim.device)), dim=1)

        # Randomize target positions
        # This creates random targets in the xy-plane within a circle of radius 1000 around the origin
        angles = torch.rand(self.num_envs, device=self.sim.device) * 2 * torch.pi
        radii = torch.sqrt(torch.rand(self.num_envs, device=self.sim.device)) * 1000  # sqrt for uniform distribution
        self.targets = torch.stack(
            (radii * torch.cos(angles), radii * torch.sin(angles), torch.zeros(self.num_envs, device=self.sim.device)),
            dim=1)
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        self.accum_joint_movement = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.sim.device)
        self.neutral_positions = torch.tensor([60/180, 60/180, -60/180, -60/180, 0, 0, 0, 0], dtype=torch.float32, device=self.sim.device)
        # Assuming self.num_envs is the number of parallel environments
        # self.neutral_positions = torch.tensor([0, 0, 0, -1, 0, 0, 0, 0], dtype=torch.float32, device=self.sim.device)
        self.neutral_positions = self.neutral_positions.repeat(self.num_envs, 1)  # Repeat for each environment

        # Initialize progress_reward as a zero tensor
        # self.heading_proj = torch.zeros(self.num_envs, device=self.sim.device)

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
        # viewer settings
        self.sim.set_camera_view([.5, .5, .5], [0.0, 0.0, 0.0])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Now 'self.robot' should be correctly initialized and available
        # Calculate error between last commanded positions and current actual positions
        self.prev_joint_positions[:, :, 0] = self.prev_joint_positions[:, :, 1]  # Move the last positions back
        self.prev_joint_positions[:, :, 1] = self.actions  # Store previous joint actions
        self.action_errors = torch.abs(self.actions - self.robot.data.joint_pos) #Compare previous action reference and current position
        self.actions = self.cfg.action_scale * actions  # Scale actions appropriately

        # Calculate movement
        joint_movements = torch.abs(self.prev_joint_positions[:, :, 1] - self.prev_joint_positions[:, :, 0])
        self.accum_joint_movement += joint_movements

    def _apply_action(self):

        # Example: If the method expects a specific shaping of the tensor:
        actions_to_apply = self.actions.unsqueeze(2)  # This would adjust shape to [2048, 8, 1]

        # Then use this reshaped actions to apply:
        self.robot.set_joint_position_target(actions_to_apply, joint_ids=self._dof_idx)

        self.motor_effort = self.robot._data.applied_torque

        self.neutral_deviation_penalty += self.cfg.pen_scale_neutral * torch.mean(
            torch.abs(self.robot.data.joint_pos-self.neutral_positions),dim=-1,)

        self.electricity_cost +=torch.mean(
            torch.abs(self.motor_effort * (self.actions - self.prev_joint_positions[:, :, 0])),
            dim=-1,
        )



    def _get_observations(self) -> dict:
        # Reshape prev_joint_positions to a 2D tensor where each row corresponds to an environment
        joint_pos_observations = self.prev_joint_positions.view(self.num_envs, -1)

        # Get the quaternion representing the root orientation of the robot
        root_quat = self.robot.data.root_quat_w.clone().view(self.num_envs, 4)

        # Extract the xy components of the heading and target vectors
        heading_xy = self.heading_vec[:, :2]  # Taking the x and y components
        target_xy = self.targets[:, :2]/1000  # Taking the x and y components
        # print(target_xy)

        # Concatenate the joint positions, quaternion, and xy components of heading and target
        observations = torch.cat((joint_pos_observations, root_quat, heading_xy, target_xy), dim=1)

        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        # velocity = self.robot.data.root_lin_vel_w  # Assuming velocity along x-direction is desired

        # Simple reward function that rewards forward movement and correct heading
        self.progress_reward = (self.potentials - self.prev_potentials)*self.cfg.rew_scale_progress

        to_target = self.targets - self.robot.data.root_pos_w
        to_target[:, 2] = 0.0
        (_, self.up_proj, self.heading_proj,
         _, _) = compute_heading_and_up(
            self.robot.data.root_quat_w, quat_conjugate(self.start_rotation).repeat((self.num_envs, 1)), to_target, self.basis_vec0, self.basis_vec1, 2
        )
        # print(self.robot.data.root_quat_w.shape)
        heading_weight_tensor = torch.ones_like(self.heading_proj) * self.cfg.rew_scale_heading
        heading_reward = torch.where(self.heading_proj > 0.8, heading_weight_tensor, self.cfg.rew_scale_heading * self.heading_proj / 0.8)

        # aligning up axis of robot and environment
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(self.up_proj > 0.93, up_reward + self.cfg.rew_scale_upright, up_reward)

        ## Calculate the variance penalty at the end of an episode
            # Groups of joints to compare
        # coxa_indices = torch.tensor([1, 3, 5, 7])  # Assuming these are indices for coxa joints
        # base_indices = torch.tensor([0, 2, 4, 6])  # Assuming these are indices for base joints
        coxa_indices = torch.tensor([4, 5, 6, 7])  # Assuming these are indices for coxa joints
        base_indices = torch.tensor([0, 1, 2, 3])  # Assuming these are indices for base joints
        # Calculate variances among groups
        std_coxa = torch.std(self.accum_joint_movement[coxa_indices])
        std_base = torch.std(self.accum_joint_movement[base_indices])
        # Combine variances into a single penalty score
        std_penalty = (std_coxa + std_base)* self.cfg.pen_scale_symmetry
        energy_penalty = self.cfg.rew_scale_energy * self.electricity_cost
        unrealistic_ref_penalty = self.cfg.pen_scale_unrealistic *torch.sum(self.action_errors)
        # print(self.neutral_deviation_penalty.shape)




        return (self.progress_reward
                + heading_reward
                + up_reward
                - energy_penalty
                - self.neutral_deviation_penalty
                # - std_penalty
                # - unrealistic_ref_penalty       #Penalizes actor for having unrealistic servo position references
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for joint limits
        to_target = self.targets - self.robot.data.root_pos_w
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height

        # Check if the episode time has exceeded the specified length
        # This assumes that your simulation steps in the environment are counted in `_pre_physics_step` or tracked elsewhere
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])

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
        self.neutral_deviation_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.electricity_cost = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.accum_joint_movement = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.sim.device)




