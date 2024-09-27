# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.a1.mdp as a1_mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

Pose_Command_Action = LocomotionVelocityRoughEnvCfg()

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort: skip


A1_PARKOUR_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
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
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.1, 0.3), platform_width=0.4),
    },
)

# Define terrain boundaries based on the terrain configuration
x_min, x_max = -A1_PARKOUR_CFG.size[0] / 2, A1_PARKOUR_CFG.size[0] / 2
y_min, y_max = -A1_PARKOUR_CFG.size[1] / 2, A1_PARKOUR_CFG.size[1] / 2
z_min, z_max = 0.0, 0.1  # Assuming the terrain is flat


@configclass
class A1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class A1CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=False,
    #     debug_vis=False,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
    #     ),
    # )

    """for the "pos_z range", refer to the terrain noise range, which generally represents the min and max height of the terrain"""
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        simple_heading=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(x_min, x_max), pos_y=(y_min, y_max)),
    )


@configclass
class A1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class A1EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=a1_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class A1RewardsCfg:
    # -- task
    # air_time = RewardTermCfg(
    #     func=a1_mdp.feet_air_time,
    #     weight=0.1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #             "command_name": "position_command",
    #             "threshold": 0.5,},
    # )
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-400.0)

    position_tracking = RewardTermCfg(
        func=a1_mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewardTermCfg(
        func=a1_mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewardTermCfg(
        func=a1_mdp.heading_command_error_abs,
        weight=-0.3,
        params={"command_name": "pose_command"},
    )
    # foot_clearance = RewardTermCfg(
    #     func=a1_mdp.foot_clearance_reward,
    #     weight=0.05,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )

    # goal_achievement = RewardTermCfg(
    #     func=a1_mdp.goal_achievement_reward,
    #     params={"goals": goals, "current_goal_index": 0, "asset_cfg": SceneEntityCfg("robot")},
    # )

    # -- penalties

    base_motion = RewardTermCfg(
        func=a1_mdp.base_motion_penalty, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    base_angular_motion = RewardTermCfg(
        func=a1_mdp.base_angular_motion_penalty, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    base_orientation = RewardTermCfg(
        func=a1_mdp.base_orientation_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    action_smoothness = RewardTermCfg(func=a1_mdp.action_smoothness_penalty, weight=-0.01)

    foot_slip = RewardTermCfg(
        func=a1_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    collision = RewardTermCfg(
        func=a1_mdp.collision,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_calf", ".*_thigh"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf", ".*_thigh"]),
            "threshold": 0.1,
        },
    )

    feet_stumble = RewardTermCfg(
        func=a1_mdp.contact_penality,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # joint_acc = RewardTermCfg(
    #     func=a1_mdp.joint_acceleration_penalty,
    #     weight=-2.5e-7,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    # joint_torques = RewardTermCfg(
    #     func=a1_mdp.joint_torques_penalty,
    #     weight=-1.0e-7,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    # joint_vel = RewardTermCfg(
    #     func=a1_mdp.joint_velocity_penalty,
    #     weight=-1.0e-2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    joint_pos_limits = RewardTermCfg(
        func=a1_mdp.joint_pos_limits,
        weight=-0.004,
    )
    hip_limit = RewardTermCfg(
        func=a1_mdp.hip_pos_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hip"),
        },
    )


@configclass
class A1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )


@configclass
class A1CurriculumCfg:

    terrain_levels = CurrTerm(func=a1_mdp.terrain_levels_vel)


@configclass
class A1EnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: A1ObservationsCfg = A1ObservationsCfg()
    actions: A1ActionsCfg = A1ActionsCfg()
    commands: A1CommandsCfg = A1CommandsCfg()

    # MDP setting
    rewards: A1RewardsCfg = A1RewardsCfg()
    terminations: A1TerminationsCfg = A1TerminationsCfg()
    events: A1EventCfg = A1EventCfg()
    curriculum: A1CurriculumCfg = A1CurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to A1
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*hip_joint": 0.0,
            ".*thigh_joint": np.pi / 4,
            ".*calf_joint": -np.pi / 2,
        }

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=A1_PARKOUR_CFG,
            max_init_terrain_level=A1_PARKOUR_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.scene.height_scanner = None


class A1EnvCfg_PLAY(A1EnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
