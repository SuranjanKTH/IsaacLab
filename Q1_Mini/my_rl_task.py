"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
#from omni.isaac.lab_assets import CARTPOLE_CFG  # replace with Q1 Mini config
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Quadruped Robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

##
# Configuration
##

Q1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Suranjan/AppData/Local/ov/pkg/isaac-lab/IsaacLab/Q1_Mini/Q1_Assembly/Q1_Mini_Test.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,  # Adjusted for realistic quadruped movement
            max_angular_velocity=10.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=10,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Position slightly above ground for initialization
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


@configclass
class Q1SceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # articulation
    q1: ArticulationCfg = Q1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["q1"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    # Design scene
    scene_cfg = Q1SceneCfg(num_envs=args_cli.num_envs, env_spacing=.50)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()