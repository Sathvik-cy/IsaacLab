
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

from .commandsCfg import Waypoint2dCommandCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv




class Waypoint2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: Waypoint2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: Waypoint2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _generate_intermediate_positions(self, current_pos, target_pos):
        """
        Generate intermediate waypoints between current position and target position.
        
        Args:
            current_pos (torch.Tensor): Current position of the robot (x, y, z).
            target_pos (torch.Tensor): Target position of the robot (x, y, z).
            num_intermediates (int): Number of intermediate positions to generate.
            
        Returns:
            torch.Tensor: A tensor containing the waypoints (including current and final positions).
        """
        # Create a linear interpolation between the current and target positions
        waypoints = torch.zeros(self.cfg.num_intermediates + 2, 3, device=self.device)
        
        # Interpolate between the current position and the target position
        for i in range(self.cfg.num_intermediates + 2):
            alpha = i / (self.cfg.num_intermediates + 1)  # Fraction along the path
            waypoints[i] = current_pos * (1 - alpha) + target_pos * alpha

        return waypoints

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]

        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.target_pos = self.pos_command_w[env_ids].clone()
        self.target_pos[:, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.target_pos[:, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.target_pos[:, 2] += self.robot.data.default_root_state[env_ids, 2]

        current_pos = self.robot.data.root_pos_w[env_ids].clone()

        # Generate 5 intermediate positions between the current and target positions
        waypoints = self._generate_intermediate_positions(current_pos, self.target_pos, num_intermediates=5)

        # Save the intermediate positions for the robot to follow
        self.waypoints = waypoints

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.target_pos - current_pos
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
    
    def _follow_waypoints(self, env_ids: Sequence[int]):
        """
        Update the robot's command to follow the next waypoint.
        """
        # Calculate the distance to the next waypoint
        distance_to_waypoint = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.waypoints[0, :2], dim=1)

        # Threshold distance to consider waypoint reached
        waypoint_threshold = 0.05  # You can adjust this value as needed

        # If robot is close to the current waypoint, move to the next one
        if distance_to_waypoint < waypoint_threshold:
            # Remove the first waypoint (current waypoint) from the list
            self.waypoints = self.waypoints[1:]

        # Update the command to follow the next waypoint
        if len(self.waypoints) > 0:
            self.pos_command_w[env_ids] = self.waypoints[0]  # Set the next waypoint as the command
        else:
            self.pos_command_w[env_ids] = self.target_pos  # Set the final target if no waypoints are left

    def _update_command(self):
        """Re-target the position command to the current root state."""

        self._follow_waypoints(env_ids=range(self.num_envs))

        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)

            # Create markers for waypoints
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)

            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.waypoint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )
        
        # Create a marker for each waypoint
        self.waypoint_visualizer.visualize(
            translations = self.waypoints[:, :3],  # Get 3D coordinates of waypoints (x, y, z)
            orientations = quat_from_euler_xyz(
                torch.zeros_like(self.waypoints[:, 0]),  # No rotation in x-axis
                torch.zeros_like(self.waypoints[:, 1]),  # No rotation in y-axis
                self.heading_command_w  # Apply heading (rotation around z-axis)
            ),
        )