import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
from pprint import pprint
from scipy.spatial.transform import Rotation
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

def compute_ik(target_P, target_R, elbow_up=True):
    if target_P is None or target_R is None:
        return None
    
    # 1. find Pw, the wrist position
    Pw = target_P - 0.065 * target_R[:,2]
    print("Pw: ", Pw)

    # 2. find q1, the first joint angle
    q1 = np.arctan2(Pw[1], Pw[0])

    # 3. find Pw' the wrist position in the xy plane
    Pw_prime = np.array([
        np.linalg.norm(Pw[0:2]), 
        Pw[2] - 0.136, 
        0])

    print("Pw_prime: ", Pw_prime)
    print("Q1: ", q1)

    # 4. find q3 
    c3 = (Pw_prime[0]**2 + Pw_prime[1]**2 - 0.1**2 - 0.107**2) / (2. * 0.1 * 0.107)
    
    # Check if c3 is within valid range [-1, 1]
    if c3 < -1 or c3 > 1:
        print(f"Invalid c3 value: {c3}, position unreachable")
        return None
        
    s3 = np.sqrt(1. - c3**2)  # Only need one solution since we'll handle elbow_up flag

    if elbow_up:
        q3 = np.arctan2(-s3, c3)  # Negative s3 for elbow up
    else:
        q3 = np.arctan2(s3, c3)   # Positive s3 for elbow down
    
    # 5. find q2
    gamma = np.arctan2(Pw_prime[1], Pw_prime[0])
    beta = np.arctan2(0.107 * np.sin(q3), 0.1 + 0.107 * np.cos(q3))
    q2 = gamma - beta
    
    # Adjust angles to match robot's joint configuration
    q2 = -q2 - 3*np.pi/2.
    q3 = -q3 - np.pi/2.

    print("Q2: ", q2)
    print("Q3: ", q3)

    # Now we know R0123 and R0123456, we can find R456
    R0123 = Rotation.from_matrix([
        [np.sin(q2+q3)*np.cos(q1), np.cos(q1)*np.cos(q2+q3), -np.sin(q1)],
        [np.sin(q1)*np.sin(q2+q3), np.sin(q1)*np.cos(q2+q3), np.cos(q1)],
        [np.cos(q2+q3), -np.sin(q2+q3), 0]
    ]).as_matrix()
    
    R456 = R0123.T @ target_R

    # Handle singular case where q5 is near 0 or pi
    if np.isclose(abs(R456[1,2]), 1.0):
        # Singular case - set q5 and choose arbitrary q4, q6
        q5 = 0.0 if R456[1,2] > 0 else np.pi
        q4 = 0.0  # Arbitrary choice
        q6 = np.arctan2(R456[0,1], R456[0,0])
    else:
        # Non-singular case
        q5 = np.arccos(R456[1,2])
        q4 = np.arctan2(R456[2,2], -R456[0,2])
        q6 = np.arctan2(R456[1,0], R456[1,1])
    
    q6 = q6 - np.pi  # Adjust for robot configuration

    print("Q4: ", q4)
    print("Q5: ", q5)
    print("Q6: ", q6)

    # check that none of the angles are NaN
    if np.isnan(q1) or np.isnan(q2) or np.isnan(q3) or np.isnan(q4) or np.isnan(q5) or np.isnan(q6):
        return None
    
    return [q1, q2, q3, q4, q5, q6]


class IK(Node):
    def __init__(self):
        super().__init__('ik_analytic_client')

        self.target_pose = None
        # subscribe to /target_pose topic
        self.create_subscription(PoseStamped, 'target_pose', self.target_pose_callback, 10)

        # publish to /joint_states topic
        self.joint_states_pub = self.create_publisher(JointState, '/joint_states', 10) 

        self.base_frame = "base"
        self.from_frame_rel = self.declare_parameter('target_frame', "target").get_parameter_value().string_value
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
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.base_frame} to {self.from_frame_rel}: {ex}')
            return

        R = Rotation.from_quat([
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w
        ]).as_matrix()

        # Compute the IK solution - try both elbow up and down
        q = compute_ik(P, R, elbow_up=True)
        if q is None:
            q = compute_ik(P, R, elbow_up=False)
            if q is None:
                self.get_logger().warn(f"IK failed for position {P}, skipping point")
                return
        
        q1, q2, q3, q4, q5, q6 = q
        self.send_joint_states(q1, q2, q3, q4, q5, q6)

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
    ik_client = IK()
    
    try:
        rclpy.spin(ik_client)
    except KeyboardInterrupt:
        pass
    
    ik_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()