import numpy as np
from scipy.spatial.transform import Rotation
from pprint import pprint

def compute_ik(target_P, target_R):
    if target_P is None or target_R is None:
        return None
    
    # 1. find Pw, the wrist position
    x, y, z = target_P
    R = target_R

    # 2. find q1, the first joint angle
    q1 = np.arctan2(-y, x)
    print("Q1: ", q1)

    # 3. find Pw' the wrist position in the xy plane
    q3 = z  # q3 is the translation along z-axis
    q2 = np.sqrt(x**2 + y**2)  # q2 is the translation along x-axis

    print("Q2: ", q2)
    print("Q3: ", q3)

    # Now we know R0123 and R0123456, we can find R456
    R0123 = np.array([
        [-np.sin(q1), np.cos(q1), 0],
        [np.cos(q1), np.cos(q1), 0],
        [0, 0, 1]
    ])
    R456 = R456 = R0123.T @ R
    print(R456)

    # ASSUMPTION: non-singular case in which r23 != +-1
   
    c5 = np.sqrt(R456[0, 0]**2 + R456[0, 1]**2)
    s5 = np.sqrt(R456[1, 2]**2 + R456[2, 2]**2)


    q4 = np.arctan2(-R456[1, 2], R456[2, 2])
    #q5 = np.arcsin(R456[0, 2])
    q5 = np.arctan2(s5, c5)
    q6 = np.arctan2(-R456[0, 1], R456[0, 0])

    print("Q4: ", q4)
    print("Q5: ", q5)
    print("Q6: ", q6)
    

    # Check that none of the angles are NaN
    if np.isnan(q1) or np.isnan(q2) or np.isnan(q3) or np.isnan(q4) or np.isnan(q5) or np.isnan(q6):
        return None
    
    return [q1, q2, q3, q4, q5, q6]


def main():
    # Example target position and orientation
    target_position = np.array([0.5, 0.5, 0.5])  # Example position (x, y, z)
    target_orientation_quat = [0, 0, 0, 1]  # Example quaternion (x, y, z, w)

    # Convert quaternion to rotation matrix
    target_orientation_matrix = Rotation.from_quat(target_orientation_quat).as_matrix()

    print("Target Position:")
    pprint(target_position)
    print("Target Orientation (Rotation Matrix):")
    pprint(target_orientation_matrix)

    # Compute the IK solution
    joint_angles = compute_ik(target_position, target_orientation_matrix)

    if joint_angles is not None:
        print("Computed Joint Angles:")
        pprint(joint_angles)
    else:
        print("Failed to compute IK solution.")


if __name__ == '__main__':
    main()