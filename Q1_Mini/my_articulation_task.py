"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with articulation on Q1 Mini.")
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
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

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



def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    q1_cfg = Q1_CFG.copy()
    q1_cfg.prim_path = "/World/Origin.*/Robot"
    q1 = Articulation(cfg=q1_cfg)

    # return the scene information
    scene_entities = {"q1": q1}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["q1"]
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
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 1
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()