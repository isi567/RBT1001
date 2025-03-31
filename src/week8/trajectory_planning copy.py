import rclpy
from rclpy.node import Node
from inverse_kinematics_analytic_mimi import compute_ik
from direct_kinematics_mimi import compute_dk
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np

class TrajectoryPlanning(Node):
    def __init__(self):
        super().__init__('traj_client')
        self.qs = None

    def plan_cartesian_trajectory(self):
        
        # Define waypoints in cartesian space (x, y, z)
        # NOTE: change at will (just make sure they are inside the robot's workspace)
        waypoints = [
            [0.1, 0.1, 0.2], #A
            [0.1, -0.1, 0.2], #B
            [-0.1, -0.1, 0.2] #C
        ]
        # Introduce 20 regularly distributed viapoints between waypoints
        viapoints = [waypoints[0]]  # Start with the first waypoint
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            for t in np.linspace(0, 1, 21)[1:]:  # Skip the first point to avoid duplication
                viapoint = [(1 - t) * start[j] + t * end[j] for j in range(3)]
                viapoints.append(viapoint)
        

        # plot them for visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints], [p[2] for p in waypoints], 'ro-', markersize=12)
        ax.plot([p[0] for p in viapoints], [p[1] for p in viapoints], [p[2] for p in viapoints], 'bo-')
        plt.show()

        # set a fixed rotation for the end effector
        R = Rotation.from_matrix([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]).as_matrix()

  

        # for each pair of viapoints, calculate the joint states using IK
        # TODO: complete the code
        # HINT: use the compute_ik function from inverse_kinematics_analytic.py
        self.qs = [
            compute_ik(viapoints[i], R)
            for i in range(len(viapoints))
        ] 
        #q = compute_ik(viapoints[0], R)
        #print(q)
        # for each joint, plot the trajectory with waypoints and via points
        for i in range(6):
            plt.plot([q[i] for q in self.qs], label='joint' + str(i + 1))
        plt.legend()
        plt.show()   
    def plan_joint_trajectory(self):
    # Define joint space waypoints (each sublist contains 6 joint angles)
        waypoints = [
        [0., 0., 0., 0., 0., 0.],      # joints at A
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  #  joints at B
        [1.5, 1.4, 1.3, 1.2, 1.1, 1.]    # joints at C
    ]
        # Introduce 20 regularly distributed viapoints between waypoints
        viapoints = [waypoints[0]]  # Start with first waypoint
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            for t in np.linspace(0, 1, 21)[1:]: # Skip the first point to avoid duplication
                viapoint = [(1 - t) * start[j] + t * end[j] for j in range(6)] # interpolate between start and end
                viapoints.append(viapoint)



        
        # plot them for visualization
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(6):
            ax.plot([0, 20, 40], [p[i] for p in waypoints],'yo-', markersize=12)
            ax.plot([p[i] for p in viapoints], 'o-', label='joint' + str(i + 1))
        ax.legend()
        plt.show()

        # for each pair of viapoints, calculate the end effector position using compute_dk function
        # TODO: complete the code
        # HINT: use the compute_dk function from direct_kinematics.py

        ##### ISSUE #####
        Ps = [
            compute_dk(viapoints[i][0], viapoints[i][1], viapoints[i][2], viapoints[i][3], viapoints[i][4], viapoints[i][5])[0]
            for i in range(len(viapoints))
        ]                                          

        # plot the trajectory in cartesian space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([p[0] for p in Ps], [p[1] for p in Ps], [p[2] for p in Ps], 'bo-')
        plt.show()


if __name__ == '__main__':
    rclpy.init()
    node = TrajectoryPlanning()
    node.plan_cartesian_trajectory()
    node.plan_joint_trajectory()
    rclpy.spin(node)
    rclpy.shutdown()
