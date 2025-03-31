import sympy as np
from sympy import Matrix, cos, sin, symbols, pprint


# returns a Homogeneous Rotation matrix, angle in radians
def HR(axis, angle):
    if axis == 'x':
        mat = Matrix([
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        mat = Matrix([
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        mat = Matrix([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis '{axis}'. Axis must be 'x', 'y', or 'z'.")
    
    return mat

# returns a Homogeneous Translation matrix, distance in meters
def HT(axis, distance):
    if axis == 'x':
        mat = Matrix([
            [1, 0, 0, distance],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        mat = Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, distance],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        mat = Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, distance],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis '{axis}'. Axis must be 'x', 'y', or 'z'.")
    
    return mat

def compute_dk(q1, q2, q3, q4, q5, q6):
    # Use matrix multiplication (@) instead of element-wise multiplication (*)
    T = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ \
        HR('x', q4) @ HR('y', q5) @ HR('z', q6)
    return T[:3, 3], T

def main():
    # Define symbolic variables
    q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
    
    # Compute direct kinematics
    position, transformation_matrix = compute_dk(q1, q2, q3, q4, q5, q6)
    
    # Print symbolic results
    print("Position:")
    pprint(position)
    print("\nTransformation Matrix:")
    pprint(transformation_matrix)

if __name__ == '__main__':
    main()