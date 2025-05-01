import rclpy
from rclpy.node import Node
from inverse_kinematics_analytic import compute_ik
# from direct_kinematics import compute_dk
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np

class TrajectoryPlanning(Node):
    def __init__(self):
        super().__init__('traj_client')

        # publish to /joint_states topic
        self.joint_states_pub = self.create_publisher(JointTrajectory, '/mecharm_controller/joint_trajectory', 10) 

        self.base_frame = "base"


    def publish_joint_trajectory(self, cartesian_trajectory):
        x_data, y_data, z_data = cartesian_trajectory

        times = x_data[0]
        x_pos = x_data[1]
        y_pos = y_data[1]
        z_pos = z_data[1]
        x_vel = x_data[2]
        y_vel = y_data[2]
        z_vel = z_data[2]

        # Create a JointTrajectory message
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.stamp = self.get_clock().now().to_msg()
        joint_trajectory.joint_names = [
            'joint1_to_base',
            'joint2_to_joint1',
            'joint3_to_joint2',
            'joint4_to_joint3',
            'joint5_to_joint4',
            'joint6_to_joint5',
        ]
        R = Rotation.from_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).as_matrix()
        # Populate the trajectory points
        for i in range(len(times)):
            # compute the joint angles using inverse kinematics
            # q = compute_ik([x_pos[i], y_pos[i], z_pos[i]], R, elbow_up=True)
            # if q is None:
            #     self.get_logger().error("Could not compute inverse kinematics for point {}.".format(i))
            #     continue
            # q1, q2, q3, q4, q5, q6 = q
            # Create a trajectory point
            point = JointTrajectoryPoint()
            # point.positions = [q1, q2, q3, q4, q5, q6]
            point.positions = [0.1*i, 0., 0., 0., 0., 0.]
            point.velocities = [0.2, 0., 0., 0., 0., 0.]
            point.time_from_start = rclpy.time.Duration(seconds=0.5 * i).to_msg()
            joint_trajectory.points.append(point)
        # Publish the trajectory
        self.joint_states_pub.publish(joint_trajectory)
        self.get_logger().info('Joint trajectory published!')


    
    def plan_cartesian_trajectory(self, waypoints=[[[0.17, 0.0, 0.24],[0.17, -0.1, 0.24]]], pdmax=0.2, times=[5]):

        # Compute the trapezoidal trajectory for each of the dimensions separately (X, Y and Z)
        # with 20 intermediate points. Repeat for each pair of waypoints
        txs, pxs, pdxs, pddxs, = [], [], [], []
        tys, pys, pdys, pddys, = [], [], [], []
        tzs, pzs, pdzs, pddzs, = [], [], [], []
        starting_time = 0.
        ticks = 20
        wp_idxs = [0]
        for init_point_idx in range(len(waypoints) - 1):
            print("Processing segment from waypoint {} to waypoint {}".format(init_point_idx, init_point_idx + 1))
            # compute the trapezoidal trajectory for each dimension
            print("Dim X")
            px0 = waypoints[init_point_idx][0] # initial point x
            pxf = waypoints[init_point_idx+1][0] # final point x
            _txs, _pxs, _pdxs,_pddxs,_newtimex = self.generate_trapezoidal_segment(times[init_point_idx], px0, pxf, pdmax, ticks=ticks)
            txs += [t+starting_time for t in _txs][:-1]
            pxs += _pxs[:-1]
            pdxs += _pdxs[:-1]
            pddxs += _pddxs[:-1]

            print("Dim Y")
            py0 = waypoints[init_point_idx][1] # initial point y
            pyf = waypoints[init_point_idx+1][1] # final point y
            _tys, _pys, _pdys,_pddys,_newtimey = self.generate_trapezoidal_segment(times[init_point_idx], py0, pyf, pdmax, ticks=ticks)
            tys += [t+starting_time for t in _tys][:-1]
            pys += _pys[:-1]
            pdys += _pdys[:-1]
            pddys += _pddys[:-1]

            print("Dim Z")
            pz0 = waypoints[init_point_idx][2] # initial point z
            pzf = waypoints[init_point_idx+1][2] # final point z
            _tzs, _pzs, _pdzs,_pddzs,_newtimez = self.generate_trapezoidal_segment(times[init_point_idx], pz0, pzf, pdmax, ticks=ticks)
            tzs += [t+starting_time for t in _tzs][:-1]
            pzs += _pzs[:-1]
            pdzs += _pdzs[:-1]
            pddzs += _pddzs[:-1]

            wp_idxs += [len(txs) - 1]
            
            starting_time += max(_newtimex, _newtimey, _newtimez)

        # fix in case any trajectory was extended in the last segment
        # this is a workaround to make the plots aligned. 
        # if txs[-1] < starting_time:
        #     txs.append(starting_time)
        #     pxs.append(pxs[-1])
        #     pdxs.append(pdxs[-1])
        #     pddxs.append(pddxs[-1])
        # if tys[-1] < starting_time:
        #     tys.append(starting_time)
        #     pys.append(pys[-1])
        #     pdys.append(pdys[-1])
        #     pddys.append(pddys[-1])
        # if tzs[-1] < starting_time:
        #     tzs.append(starting_time)
        #     pzs.append(pzs[-1])
        #     pdzs.append(pdzs[-1])
        #     pddzs.append(pddzs[-1])

        # plot all the position and velocity trajectories for the three dimensions
        fig, axs = plt.subplots(3, 3, figsize=(12, 8))
        fig.suptitle('Trapezoidal Trajectory')
        axs[0, 0].set_title('X')
        axs[0, 0].set_xlabel('time [s]')
        axs[0, 0].set_ylabel('position [m]')
        axs[0, 1].set_title('X')
        axs[0, 1].set_xlabel('time [s]')
        axs[0, 2].set_xlabel('time [s]')
        axs[0, 1].set_ylabel('velocity [m/s]')
        axs[0, 2].set_ylabel('acceleration [m/s]')
        axs[1, 0].set_title('Y')
        axs[1, 0].set_xlabel('time [s]')
        axs[1, 0].set_ylabel('position [m]')
        axs[1, 1].set_title('Y')
        axs[1, 1].set_xlabel('time [s]')
        axs[1, 2].set_xlabel('time [s]')
        axs[1, 1].set_ylabel('velocity [m/s]')
        axs[1, 2].set_ylabel('acceleration [m/s]')
        axs[2, 0].set_title('Z')
        axs[2, 0].set_xlabel('time [s]')
        axs[2, 0].set_ylabel('position [m]')
        axs[2, 1].set_title('Z')
        axs[2, 1].set_xlabel('time [s]')
        axs[2, 2].set_xlabel('time [s]')
        axs[2, 1].set_ylabel('velocity [m/s]')
        axs[2, 2].set_ylabel('acceleration [m/s]')
        axs[0, 0].plot(txs, pxs,'ro-', markersize=2)
        axs[0, 1].plot(txs, pdxs, 'bo-', markersize=2)
        axs[0, 2].plot(txs, pddxs, 'yo-', markersize=2)
        axs[1, 0].plot(tys, pys, 'ro-', markersize=2)
        axs[1, 1].plot(tys, pdys,'bo-', markersize=2)
        axs[1, 2].plot(tys, pddys,'yo-', markersize=2)
        axs[2, 0].plot(tzs, pzs, 'ro-', markersize=2)
        axs[2, 1].plot(tzs, pdzs, 'bo-', markersize=2)
        axs[2, 2].plot(tzs, pddzs, 'yo-', markersize=2)
        # add the waypoints to position plots and corresponding vertical lines to velocities plots
        for i in range(len(waypoints)):
            axs[0, 0].plot(txs[wp_idxs[i]], waypoints[i][0], 'go', markersize=7)
            axs[1, 0].plot(tys[wp_idxs[i]], waypoints[i][1], 'go', markersize=7)
            axs[2, 0].plot(tzs[wp_idxs[i]], waypoints[i][2], 'go', markersize=7)
            axs[0, 1].axvline(x=txs[wp_idxs[i]], color='g', linestyle='--')
            axs[1, 1].axvline(x=tys[wp_idxs[i]], color='g', linestyle='--')
            axs[2, 1].axvline(x=tzs[wp_idxs[i]], color='g', linestyle='--')
            axs[0, 2].axvline(x=txs[wp_idxs[i]], color='g', linestyle='--')
            axs[1, 2].axvline(x=tys[wp_idxs[i]], color='g', linestyle='--')
            axs[2, 2].axvline(x=tzs[wp_idxs[i]], color='g', linestyle='--')
        
        plt.tight_layout()
        plt.show()

        return (txs, pxs, pdxs, pddxs), \
               (tys, pys, pdys, pddys), \
               (tzs, pzs, pdzs, pddzs)

    # Linear Segments with Parabolic Blends
    # Computes the lspb along one dimention (X, Y or Z)
    #  tf:    the final time [secs] - int
    #  p0:    initial position [m] - float
    #  pf:    final position [m] - float
    #  pdmax: maximum velocity [m/s] - float
    #  ticks: number of subdivision points in the trajectory - int
    # Returns:
    #  Ts: corresponding list of times - list of float
    #  Ps: list of position values defined along the trajectory - list of float
    #  Pds: list of velocities - list of float
    def generate_trapezoidal_segment(self, tf, p0, pf, pdmax, ticks):
        
        print("[LSPB] Input data\n tf:{}, p0:{}, pf:{}, pdmax:{}, ticks:{}".format(tf, p0, pf, pdmax, ticks))

        # list of times
        ts = [0.] + [float(i+1)*float(tf)/ticks for i in range(ticks)]

        if abs(p0 - pf) <= 1e-6:
            print("[LSPB] \t p0 and pf are equal")
            return ts, [p0]*(ticks+1), [0.]*(ticks+1), [0.]*(ticks+1), tf

        # adjust the velocity sign based on the traj direction (i.e. velocity sign might need to be negative)
        pdmax = abs(pdmax) * np.sign(pf - p0)

        trapezoidal = True
        if abs(pdmax) > 2*abs(pf-p0)/tf:
            print("We cannot reach max velocity in the given time, using triangular profile")
            trapezoidal = False

        if trapezoidal:
            # find tb
            tb = abs((p0 - pf + pdmax* tf) / pdmax)
            print("[LSPB] \t tb:{}".format(tb))


            if abs(pdmax) < abs(pf-p0)/tf:
                print("We cannot reach the goal in the given time within the velocity limits, adjusting time")
                tf = (tb * 2) + (pf - p0 - tb*pdmax) / pdmax 
                ts = [0.] + [float(i+1)*float(tf)/ticks for i in range(ticks)]
                print("[LSPB] \t new tf:{}".format(tf))

            # find max acceleration
            pddmax = pdmax / tb
            print("[LSPB] \t pddmax:{}".format(pddmax))

            # list of angles
            ps = [0.0] * len(ts)
            # list of velocities
            pds = [0.0] * len(ts)
            # list of accelerations
            pdds = [0.0] * len(ts)
            print("[LSPB] \t ts:{}".format(ts))
            for i, t in enumerate(ts):
                if t <= tb:
                    ps[i] = p0 + (0.5 * pddmax * t**2)
                    pds[i] = pddmax * t
                    pdds[i] = pddmax
                elif t <= tf - tb:
                    ps[i] = p0 + (0.5 * pddmax * tb**2) + pdmax * (t - tb)
                    pds[i] = pdmax
                    pdds[i] = 0.
                else:
                    ps[i] = pf - (0.5 * pddmax * (tf - t)**2)
                    pds[i] = pdmax - (pddmax * (t - (tf - tb)))
                    pdds[i] = - pddmax
        else:
            # find tb
            tb = tf / 2
            print("[LSPB] \t tb:{}".format(tb))

            # find max acceleration
            pddmax = pdmax / tb
            print("[LSPB] \t pddmax:{}".format(pddmax))

            # list of angles
            ps = [0.0] * len(ts)
            # list of velocities
            pds = [0.0] * len(ts)
            # list of accelerations
            pdds = [0.0] * len(ts)
            print("[LSPB] \t ts:{}".format(ts))
            for i, t in enumerate(ts):
                if t <= tb:
                    ps[i] = p0 + (0.5 * pddmax * t**2)
                    pds[i] = pddmax * t
                    pdds[i] = pddmax
                else:
                    ps[i] = pf - (0.5 * pddmax * (t - tf)**2) 
                    pds[i] = pddmax * (tf -  t)
                    pdds[i] = - pddmax

        print("[LSPB] \t ps:{}".format(ps))
        print("[LSPB] \t pds:{}".format(pds))

        return ts, ps, pds, pdds, tf



if __name__ == '__main__':
    rclpy.init()
    node = TrajectoryPlanning()
    cartesian_traj = node.plan_cartesian_trajectory()
    node.publish_joint_trajectory(cartesian_traj)

    rclpy.spin(node)
    rclpy.shutdown()
