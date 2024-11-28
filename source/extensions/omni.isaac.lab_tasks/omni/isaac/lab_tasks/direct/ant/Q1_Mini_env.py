# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from omni.isaac.lab.actuators import ImplicitActuatorCfg

gear = 1.5
Q1_effort_limit = 0.0       #Doesn't seem to have any effect
Q1_velocity_limit = 0.0     #Doesn't seem to have any effect
Q1_damping = 5.0*gear            #kg/s

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
        pos=(0.0, 0.0, 0.1),  # Position slightly above ground for initialization
        joint_pos={
            "BASE_Revolute_1": 0.0,
            "BASE_Revolute_3": 0.0,
            "BASE_Revolute_5": 0.0,
            "BASE_Revolute_7": 0.0,
            "Coxa_BL_Revolute_21": 0.0,
            "Coxa_FR_Revolute_4": 0.0,
            "Coxa_FL_Revolute_6": 0.0,
            "Coxa_BR_Revolute_8": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=50.0,
        ),
    },
)


## Configuration for the Q1 Mini Robot ##
@configclass
class Q1MiniEnvCfg(DirectRLEnvCfg):
    # Env parameters
    episode_length_s = 5.0
    decimation = 2
    action_scale = 0.5
    action_space = 8  # Adjust according to your robot
    observation_space = 36
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=.50, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = Q1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [gear,gear,gear,gear,gear,gear,gear,gear]

    # Reward parameters
    heading_weight: float = 0.5
    up_weight: float = 0.001

    energy_cost_scale: float = 0.0
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.01

    death_cost: float = -2.0
    termination_height: float = 0.028

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


class Q1MiniEnv(LocomotionEnv):
    cfg: Q1MiniEnvCfg

    def __init__(self, cfg: Q1MiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


