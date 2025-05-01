#!/usr/bin/env python3

import rclpy
import time
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

# Import existing functionality
from trapezoidal_trajectory import TrajectoryPlanning 
from inverse_kinematics_analytic import compute_ik    

#################################################
# STUDENT CONFIGURATION SECTION - MODIFY THIS   #
#################################################

# Define waypoints in cartesian space (x, y, z)
# NOTE: Make sure they are inside the robot's workspace
CARTESIAN_WAYPOINTS = [
            [0.1, 0.1, 0.0],
            [0.1, -0.7, 0.3],
            [-0.2, -0.11, 0.2],
            [0.1, 0.1, 0.0],# Return to starting position
]

# If you want to set joint angles directly, use this list instead
# Format: [joint1, joint2, joint3, joint4, joint5, joint6]
JOINT_WAYPOINTS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],             # Home position
    [0.5, 0.4, 0.2, 0.3, 0.3, 0.5],             # Position 1
    [-0.3, 0.8, -0.5, -0.2, 0.5, 1.0],          # Position 2
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]              # Back to home position
]

# Choose whether to use Cartesian waypoints or joint waypoints
USE_CARTESIAN = False

# Trapezoidal trajectory parameters
MAX_CARTESIAN_VELOCITY = 1.0  # m/s
SEGMENT_TIME = 2.0  # seconds per segment
TICKS_PER_SEGMENT = 60  # number of points per segment

# Simulation parameters
ACTION_SERVER = '/joint_trajectory_controller/follow_joint_trajectory'
JOINT_STATES_TOPIC = '/joint_states'

# Joint names for the mecharm robot
JOINT_NAMES = [
    "joint1_to_base",
    "joint2_to_joint1", 
    "joint3_to_joint2", 
    "joint4_to_joint3", 
    "joint5_to_joint4", 
    "joint6_to_joint5"
]

#################################################
# TRAJECTORY EXECUTION CLASS                    #
#################################################

class TrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('trajectory_executor_node')
        
        # Create action client for trajectory control
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            ACTION_SERVER
        )
        
        # Subscribe to joint states to monitor trajectory execution
        self.joint_state_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self.joint_state_callback,
            10
        )
        
        # Initialize variables for tracking progress
        self.current_positions = [0.0] * len(JOINT_NAMES)
        self.trajectory_sent = False
        
        # Create a trajectory planner instance from the existing class
        self.trajectory_planner = TrajectoryPlanning()
        
        self.get_logger().info('Trajectory Executor initialized')
        
        # Wait a moment for connections to establish
        time.sleep(1.0)
    
    def joint_state_callback(self, msg):
        """Update current joint positions when joint states are received"""
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_positions[i] = msg.position[idx]
    
    def seconds_to_duration_msg(self, seconds):
        """Convert floating point seconds to a Duration message"""
        return DurationMsg(
            sec=int(seconds),
            nanosec=int((seconds - int(seconds)) * 1e9)
        )
    
    def execute_joint_trajectory(self, joint_positions, times):
        """
        Execute a trajectory with the specified joint positions at the specified times
        
        Args:
            joint_positions: List of joint angle lists [q1, q2, q3, q4, q5, q6]
            times: List of time points
        """
        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available after 5 seconds')
            return False
            
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = self.seconds_to_duration_msg(seconds=1.0)
        # goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal_msg.trajectory.joint_names = JOINT_NAMES
        
        # Add trajectory points
        for i in range(len(joint_positions)):
            point = JointTrajectoryPoint()
            point.positions = joint_positions[i]
            point.velocities = [0.0] * len(JOINT_NAMES)  # Could set velocities if needed
            point.time_from_start = self.seconds_to_duration_msg(times[i])
            goal_msg.trajectory.points.append(point)
        
        # Send goal via action client
        self.get_logger().info(f'Sending trajectory with {len(goal_msg.trajectory.points)} points')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.trajectory_sent = True
        return True
    
    def goal_response_callback(self, future):
        """Callback when goal is accepted or rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by action server')
            return
        
        self.get_logger().info('Goal accepted by action server')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """Callback when action is completed"""
        result = future.result().result
        self.get_logger().info(f'Action completed with error code: {result.error_code}')
        
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Trajectory executed successfully')
        else:
            self.get_logger().warning(f'Trajectory execution failed with error code {result.error_code}')
            
    def feedback_callback(self, feedback_msg):
        """Process feedback during trajectory execution"""
        feedback = feedback_msg.feedback
        # Logging a brief feedback without overwhelming the console
        if len(feedback.actual.positions) > 0:
            self.get_logger().info(f'Executing trajectory... Current position: {[round(p, 2) for p in feedback.actual.positions]}')
    
    def execute_cartesian_trajectory(self, cartesian_waypoints):
        """
        Plan and execute a trajectory through the specified Cartesian waypoints
        
        Args:
            cartesian_waypoints: List of [x, y, z] positions
        """
        self.get_logger().info('Planning Cartesian trajectory...')
        
        # Use the existing TrajectoryPlanning implementation to generate paths
        # Extract the position and time data from the returned trajectories
        x_traj, y_traj, z_traj = self.trajectory_planner.plan_cartesian_trajectory(
            cartesian_waypoints,
            MAX_CARTESIAN_VELOCITY,
            [SEGMENT_TIME] * (len(cartesian_waypoints) -1))
        
        # Extract times and positions from trajectory data
        times = x_traj[0]  # We'll use the x trajectory times for all dimensions
        positions = []
        
        # Combine x, y, z positions at each time point
        for i in range(len(times)):
            positions.append([
                x_traj[1][i],  # x position
                y_traj[1][i],  # y position
                z_traj[1][i]   # z position
            ])
        
        # Convert Cartesian trajectory to joint trajectory using inverse kinematics
        joint_positions = []
        
        # Default orientation for end-effector (identity matrix)
        R = np.eye(3)
        
        for position in positions:
            # Calculate joint angles using the imported compute_ik function
            joint_angles = compute_ik(position, R, True)
            
            # If inverse kinematics fails for a point, skip it
            if joint_angles is None:
                self.get_logger().warn(f'IK failed for position {position}, skipping point')
                continue
                
            joint_positions.append(joint_angles)
        
        if not joint_positions:
            self.get_logger().error('No valid joint positions could be calculated')
            return False
        
        # Execute the joint trajectory
        return self.execute_joint_trajectory(joint_positions, times[:len(joint_positions)])

#################################################
# MAIN FUNCTION                                 #
#################################################

def main():
    rclpy.init()
    executor = TrajectoryExecutor()
    
    try:
        success = False
        
        if USE_CARTESIAN:
            
            # Execute trajectory using Cartesian waypoints
            executor.get_logger().info('Executing Cartesian trajectory')
            success = executor.execute_cartesian_trajectory(CARTESIAN_WAYPOINTS)
        else:
            # Execute trajectory using joint waypoints directly
            executor.get_logger().info('Executing joint trajectory')
            
            # Create time points (evenly spaced)
            total_time = SEGMENT_TIME * (len(JOINT_WAYPOINTS) - 1)
            times = [i * total_time / (len(JOINT_WAYPOINTS) - 1) for i in range(len(JOINT_WAYPOINTS))]
            
            success = executor.execute_joint_trajectory(JOINT_WAYPOINTS, times)
            executor.get_logger().info(f'Joint trajectory sent successfully: {success}')
        
        if success:
            # Run until execution is complete or interrupted
            executor.get_logger().info('Trajectory sent, waiting for completion...')
            rclpy.spin(executor)
        else:
            executor.get_logger().error('Failed to send trajectory')
        
    except KeyboardInterrupt:
        executor.get_logger().info('Interrupted by user')
    finally:
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
