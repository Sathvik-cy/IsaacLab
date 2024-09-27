# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test.py")
AppLauncher.add_app_launcher_args(parser)

# parse the arguments

args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import torch
from dataclasses import MISSING

import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Define terrain configuration
    terrain = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=9,
        num_cols=21,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
            ),
        },
    )

    # Generate the terrain

    # Add the robot to the environment
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["UnitreeA1"]
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

            def generate_random_goals(num_goals, x_min, x_max, y_min, y_max, z_min, z_max):
                goals = []
                for _ in range(num_goals):
                    x = np.random.uniform(x_min, x_max)
                    y = np.random.uniform(y_min, y_max)
                    z = np.random.uniform(z_min, z_max)
                    weight = np.random.uniform(0.5, 2.0)  # Assign random weights between 0.5 and 2.0
                    goals.append({"coordinates": (x, y, z), "weight": weight})
                return goals

            size = [8.0, 8.0]
            # Define terrain boundaries based on the terrain configuration
            x_min, x_max = -size[0] / 2, size[0] / 2
            y_min, y_max = -size[1] / 2, size[1] / 2
            z_min, z_max = 0.0, 0.1  # Assuming the terrain is flat

            # Generate random goals
            num_goals = 10  # Number of goals to generate
            goals = generate_random_goals(num_goals, x_min, x_max, y_min, y_max, z_min, z_max)

            coordinates_list = [goal["coordinates"] for goal in goals]

            for i, origin in enumerate(coordinates_list):
                prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
            # Defomrable Object

            goal_object = []
            for i, origin in enumerate(coordinates_list):
                cfg = DeformableObjectCfg(
                    prim_path=f"/World/Origin{i}/Cube",
                    spawn=sim_utils.MeshCuboidCfg(
                        size=(0.2, 0.2, 0.2),
                        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                        physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
                    ),
                    init_state=DeformableObjectCfg.InitialStateCfg(),
                    debug_vis=True,
                )

                goal_object.append(DeformableObject(cfg=cfg))
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
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=2.0)
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
    # Function to generate random goals
