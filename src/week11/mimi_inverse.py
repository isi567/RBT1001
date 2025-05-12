import numpy as np
from scipy.spatial.transform import Rotation

# Define the IK function (assuming it's in the same file or imported)
def compute_ik(position, target_R=None, elbow_up=True):

    x, y, z = position

    # Compute q1
    q1 = np.arctan2(y, x)

    # Compute wrist position (Pw_prime)
    Pw_prime = np.array([x, y, z]) - np.array([0, 0, 0.1])  # Adjust for link offsets

    # Compute q3
    c3 = (Pw_prime[0]**2 + Pw_prime[1]**2 - 0.1**2 - 0.107**2) / (2 * 0.1 * 0.107)
    c3 = np.clip(c3, -1, 1)  # Clamp c3 to the range [-1, 1] to avoid invalid sqrt
    s3p = np.sqrt(1 - c3**2)
    q3 = np.arctan2(s3p, c3)

    # Compute q2
    q2 = np.arctan2(Pw_prime[1], Pw_prime[0]) - np.arctan2(0.107 * np.sin(q3), 0.1 + 0.107 * np.cos(q3))
    q2 = -q2 - 3 * np.pi / 2.
    q3 = -q3 - np.pi / 2.

    # Compute R0123 (optional, for orientation)
    R0123 = Rotation.from_matrix([
        [np.sin(q2 + q3) * np.cos(q1), np.cos(q1) * np.cos(q2 + q3), -np.sin(q1)],
        [np.sin(q1) * np.sin(q2 + q3), np.sin(q1) * np.cos(q2 + q3), np.cos(q1)],
        [np.cos(q2 + q3), -np.sin(q2 + q3), 0]
    ]).as_matrix()

    return [q1, q2, q3]

# Define Cartesian waypoints
CARTESIAN_WAYPOINTS = [
    [0.1, 0.1, 0.0],  # Starting position
    [0.3, -0.45, 0.25], # Move to the target position
    [0.0, 0.45, 0.25], # Move to the box's position
    [0.1, 0.1, 0.0],  # Return to starting position
    [0.0, 0.0, 0.0]   # Final position (home)
]

# Compute IK for each waypoint
JOINT_WAYPOINTS = []
for waypoint in CARTESIAN_WAYPOINTS:
    try:
        joint_angles = compute_ik(waypoint)
        JOINT_WAYPOINTS.append(joint_angles)
    except Exception as e:
        print(f"Failed to compute IK for waypoint {waypoint}: {e}")

# Print the computed joint waypoints
print("Computed Joint Waypoints:")
for i, joint_angles in enumerate(JOINT_WAYPOINTS):
    print(f"Waypoint {i + 1}: {joint_angles}")