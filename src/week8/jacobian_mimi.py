import numpy as np

# returns a Homogeneous Rotation matrix, angle in radians
def HR(axis, angle):
    if axis == 'x':
        mat = [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'y':
        mat = [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'z':
        mat = [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    return np.array(mat)

# returns a Homogeneous Translation matrix, distance in meters
def HT(axis, distance):
    if axis == 'x':
        mat = [
            [1, 0, 0, distance],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'y':
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, distance],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'z':
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, distance],
            [0, 0, 0, 1]
        ]
    return np.array(mat)

def dk(q1, q2, q3, q4, q5, q6):
    T = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ \
        HR('x', q4) @ HR('y', q5) @ HR('z', q6)
    return T

def full_jacobian(q1, q2, q3, q4, q5, q6):
    J = np.zeros((6, 6))
    J[:, 0] = J1(q1, q2, q3, q4, q5, q6)
    J[:, 1] = J2(q1, q2, q3, q4, q5, q6)
    J[:, 2] = J3(q1, q2, q3, q4, q5, q6)
    J[:, 3] = J4(q1, q2, q3, q4, q5, q6)
    J[:, 4] = J5(q1, q2, q3, q4, q5, q6)
    J[:, 5] = J6(q1, q2, q3, q4, q5, q6)
    return J

def J1(q1, q2, q3, q4, q5, q6):
    jw1 = np.array([0, 0, 1])
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    jv1 = np.cross(jw1, P6)
    j1 = np.concatenate((jv1, jw1), axis=0)
    return j1

def J2(q1, q2, q3, q4, q5, q6):
    # Joint 2 is prismatic along x-axis after q1 rotation
    R01 = HR('z', q1)[:3, :3]
    jv2 = R01 @ np.array([1, 0, 0])  # x-axis of frame 1
    jw2 = np.array([0, 0, 0])  # prismatic joint has no angular component
    
    # Get position of end effector relative to joint 2
    T01 = HR('z', q1) @ HT('x', q2)
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P2 = T01[:3, 3]
    jv2 = np.cross(np.array([0, 0, 0]), (P6 - P2)) + jv2  # For prismatic, just the direction
    
    j2 = np.concatenate((jv2, jw2), axis=0)
    return j2

def J3(q1, q2, q3, q4, q5, q6):
    # Joint 3 is prismatic along z-axis after q1 and q2 transformations
    T01 = HR('z', q1) @ HT('x', q2)
    R02 = T01[:3, :3] @ HR('z', q3)[:3, :3]
    jv3 = R02 @ np.array([0, 0, 1])  # z-axis of frame 2
    jw3 = np.array([0, 0, 0])  # prismatic joint has no angular component
    
    # Get position of end effector relative to joint 3
    T02 = HR('z', q1) @ HT('x', q2) @ HT('z', q3)
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P3 = T02[:3, 3]
    jv3 = np.cross(np.array([0, 0, 0]), (P6 - P3)) + jv3  # For prismatic, just the direction
    
    j3 = np.concatenate((jv3, jw3), axis=0)
    return j3

def J4(q1, q2, q3, q4, q5, q6):
    # Joint 4 is revolute about x-axis
    T03 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4)
    R03 = T03[:3, :3]
    jw4 = R03 @ np.array([1, 0, 0])  # x-axis of frame 3
    
    # Get position of end effector relative to joint 4
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P4 = T03[:3, 3]
    jv4 = np.cross(jw4, (P6 - P4))
    
    j4 = np.concatenate((jv4, jw4), axis=0)
    return j4

def J5(q1, q2, q3, q4, q5, q6):
    # Joint 5 is revolute about y-axis
    T04 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4) @ HR('y', q5)
    R04 = T04[:3, :3]
    jw5 = R04 @ np.array([0, 1, 0])  # y-axis of frame 4
    
    # Get position of end effector relative to joint 5
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P5 = T04[:3, 3]
    jv5 = np.cross(jw5, (P6 - P5))
    
    j5 = np.concatenate((jv5, jw5), axis=0)
    return j5

def J6(q1, q2, q3, q4, q5, q6):
    # Joint 6 is revolute about z-axis
    T05 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4) @ HR('y', q5) @ HR('z', q6)
    R05 = T05[:3, :3]
    jw6 = R05 @ np.array([0, 0, 1])  # z-axis of frame 5
    
    # Get position of end effector relative to joint 6
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P6_joint = T05[:3, 3]  # For joint 6, the position is the same as end effector
    jv6 = np.cross(jw6, (P6 - P6_joint))  # Should be [0,0,0] since P6 == P6_joint
    
    j6 = np.concatenate((jv6, jw6), axis=0)
    return j6