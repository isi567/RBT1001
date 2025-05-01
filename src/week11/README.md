# Week 11: Simulated Trajectories - Student Exploration Guide

This guide will help you explore and experiment with the `simulated_trajectories.py` script for trajectory planning and execution in a simulated robotic environment.

## 1. Overview

The script demonstrates how to:
- Define waypoints in both Cartesian and joint space.
- Plan and execute trajectories using ROS 2 action servers.
- Convert Cartesian paths to joint trajectories using inverse kinematics.

## 2. Getting Started

1. **Open the Dev Container**  
   Make sure you are working inside the provided dev container.

2. **Install Dependencies**  
   Ensure all Python and ROS 2 dependencies are installed.

3. **Run the Script**  
   In the terminal, execute:
   ```bash
   python3 simulated_trajectories.py
   ```

## 3. Key Sections to Explore

### A. Waypoint Configuration

- **Cartesian Waypoints:**  
  Edit the `CARTESIAN_WAYPOINTS` list to define new end-effector positions in 3D space.
- **Joint Waypoints:**  
  Edit the `JOINT_WAYPOINTS` list to specify direct joint angles.

- **Switch Modes:**  
  Set `USE_CARTESIAN = True` to use Cartesian waypoints, or `False` to use joint waypoints.

### B. Trajectory Parameters

- Adjust `MAX_CARTESIAN_VELOCITY`, `SEGMENT_TIME`, and `TICKS_PER_SEGMENT` to see how they affect the trajectory.

### C. Execution and Feedback

- Observe the console output for feedback on trajectory execution.
- Try interrupting the script with `Ctrl+C` to see how it handles shutdown.

## 4. Suggested Experiments

- **Modify Waypoints:**  
  Try adding, removing, or changing waypoints. Observe how the robot's motion changes.
- **Test IK Failures:**  
  Set a Cartesian waypoint outside the robot's workspace and see how the script handles it.
- **Change Timing:**  
  Increase or decrease `SEGMENT_TIME` and note the effect on speed and smoothness.

## 5. Troubleshooting

- If the action server is not available, check your ROS 2 simulation environment.
- If inverse kinematics fails, ensure your waypoints are reachable.

## 6. Further Exploration

- Implement velocity or acceleration profiles for smoother motion.
- Visualize the planned trajectory using matplotlib or RViz.
- Integrate obstacle avoidance or dynamic re-planning.

---

**Happy experimenting!**
