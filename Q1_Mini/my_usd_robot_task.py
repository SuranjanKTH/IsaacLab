import argparse
from omni.isaac.kit import SimulationApp

CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
parser = argparse.ArgumentParser("Print Articulation Joints Example")
parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
args, unknown = parser.parse_known_args()
CONFIG["headless"] = args.headless
simulation_app = SimulationApp(launch_config=CONFIG)

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView

# Path to the USD file containing your robot
USD_PATH = "C:/Users/Suranjan/AppData/Local/ov/pkg/isaac-lab/IsaacLab/Q1_Mini/Q1_Assembly/Q1_Mini_Test.usd"  # Replace with the actual path to your USD file
ROBOT_PRIM_PATH = "/World/Q1_Assembly"  # Path to your robot in the USD hierarchy

# Open the USD stage
from omni.usd import get_context
get_context().open_stage(USD_PATH)

# Wait for stage to load
print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading
while is_stage_loading():
    simulation_app.update()
print("Stage loaded.")

# Initialize the world
world = World(stage_units_in_meters=1.0)

# Create ArticulationView for the robot
robot_view = ArticulationView(prim_paths_expr=ROBOT_PRIM_PATH, name="quadruped")
world.scene.add(robot_view)
world.reset()

# Print joint information
print("\n[INFO] Printing articulation joint details:")
robot_view.initialize(world.physics_sim_view)  # Ensure the articulation is initialized
joint_count = robot_view.get_joints().shape[0]

if joint_count > 0:
    print(f"\n[INFO] Found {joint_count} joints in the articulation:")
    for idx, joint_path in enumerate(robot_view.get_prim_paths(), start=1):
        print(f"  {idx}. Joint Path: {joint_path}")
else:
    print("[WARNING] No joints found in the articulation!")

# Close the simulation
simulation_app.close()