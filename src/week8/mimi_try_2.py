import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class IKClient(Node):
    def __init__(self):
        super().__init__('ik_analytic_client')

        self.target_pose = None
        # subscribe to /target_pose topic
        self.create_subscription(PoseStamped, 'target_pose', self.target_pose_callback, 10)

        # publish to /joint_states topic
        self.joint_states_pub = self.create_publisher(JointState, '/joint_states', 10) 

        self.base_frame = "base"

        # Declare and acquire `target_frame` parameter
        self.from_frame_rel = self.declare_parameter(
          'target_frame', "target").get_parameter_value().string_value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        

        self.get_logger().info('IK client ready, waiting for target poses...')

    def target_pose_callback(self, msg):
        self.target_pose = msg

        P = np.array([
            self.target_pose.pose.position.x,
            self.target_pose.pose.position.y,
            self.target_pose.pose.position.z
        ])

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.from_frame_rel,
                rclpy.time.Time())
            self.get_logger().info(
                f'Transform {self.base_frame} to {self.from_frame_rel} is: {t}')
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.base_frame} to {self.from_frame_rel}: {ex}')
        else:
            R = Rotation.from_quat([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ]).as_matrix()

            print("Target Position (P):")
            print(P)
            print("Target Rotation Matrix (R):")
            print(R)

            # Compute the IK solution
            self.compute_ik(P, R)

    def compute_ik(self, current_P, current_R):
        if self.target_pose is None:
            return
        
        # Extract position and orientation
        x, y, z = current_P
        R = current_R

        # Placeholder for joint angles
        q1, q2, q3, q4, q5, q6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Calculate q1
        q1 = np.arctan2(y, x)

        # Calculate q2 and q3 using geometric approach
       
        s = z
        q3 = s  #  q3 is the translation along z-axis
        q2 = x/np.cos(q1) # q2 is the translation along x-axis

        # Calculate q4, q5, q6 based on orientation
        R0_3 = np.array([
            [np.cos(q1), -np.sin(q1), 0],
            [np.sin(q1), np.cos(q1), 0],
            [0, 0, 1]
        ])
        R3_6 = np.dot(np.linalg.inv(R0_3), R)

        print("Intermediate Rotation Matrix (R0_3):")
        print(R0_3)
        print("Wrist Rotation Matrix (R3_6):")
        print(R3_6)

        q4 = np.arctan2(-R3_6[1, 2], R3_6[2, 2])
        q5 = np.arcsin(R3_6[0, 2])
        q6 = np.arctan2(-R3_6[0, 1], R3_6[0, 0])

        print("Joint Angles:")
        print(f"q1: {q1}, q2: {q2}, q3: {q3}, q4: {q4}, q5: {q5}, q6: {q6}")

        # check that none of the angles are NaN
        if np.isnan(q1) or np.isnan(q2) or np.isnan(q3) or np.isnan(q4) or np.isnan(q5) or np.isnan(q6):
            return
        
        # Publish the joint states
        self.send_joint_states(q1, q2, q3, q4, q5, q6)

    # Publish the joint states
    def send_joint_states(self, q1, q2, q3, q4, q5, q6):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'joint1_to_base',
            'joint2_to_joint1',
            'joint3_to_joint2',
            'joint4_to_joint3',
            'joint5_to_joint4',
            'joint6_to_joint5',
        ]
        msg.position = [q1, q2, q3, q4, q5, q6]
        self.joint_states_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    ik_client = IKClient()
    
    try:
        rclpy.spin(ik_client)
    except KeyboardInterrupt:
        pass
    
    ik_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()