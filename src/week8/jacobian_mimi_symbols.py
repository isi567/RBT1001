import sympy as sp

# Define symbolic variables
q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

# Homogeneous Rotation matrix
def HR(axis, angle):
    c, s = sp.cos(angle), sp.sin(angle)
    if axis == 'x':
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        return sp.Matrix([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        return sp.Matrix([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

# Homogeneous Translation matrix
def HT(axis, distance):
    if axis == 'x':
        return sp.Matrix([
            [1, 0, 0, distance],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, distance],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, distance],
            [0, 0, 0, 1]
        ])

# Forward kinematics
def dk(q1, q2, q3, q4, q5, q6):
    return HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ \
           HR('x', q4) @ HR('y', q5) @ HR('z', q6)

# Jacobian columns
def J1(q1, q2, q3, q4, q5, q6):
    jw1 = sp.Matrix([0, 0, 1])
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    jv1 = jw1.cross(P6)
    return sp.Matrix.vstack(jv1, jw1)

def J2(q1, q2, q3, q4, q5, q6):
    R01 = HR('z', q1)[:3, :3]
    jv2 = R01 @ sp.Matrix([1, 0, 0])
    jw2 = sp.Matrix([0, 0, 0])
    return sp.Matrix.vstack(jv2, jw2)

def J3(q1, q2, q3, q4, q5, q6):
    T01 = HR('z', q1) @ HT('x', q2)
    R02 = T01[:3, :3] @ HR('z', q3)[:3, :3]
    jv3 = R02 @ sp.Matrix([0, 0, 1])
    jw3 = sp.Matrix([0, 0, 0])
    return sp.Matrix.vstack(jv3, jw3)

def J4(q1, q2, q3, q4, q5, q6):
    T03 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4)
    R03 = T03[:3, :3]
    jw4 = R03 @ sp.Matrix([1, 0, 0])
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P4 = T03[:3, 3]
    jv4 = jw4.cross(P6 - P4)
    return sp.Matrix.vstack(jv4, jw4)

def J5(q1, q2, q3, q4, q5, q6):
    T04 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4) @ HR('y', q5)
    R04 = T04[:3, :3]
    jw5 = R04 @ sp.Matrix([0, 1, 0])
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P5 = T04[:3, 3]
    jv5 = jw5.cross(P6 - P5)
    return sp.Matrix.vstack(jv5, jw5)

def J6(q1, q2, q3, q4, q5, q6):
    T05 = HR('z', q1) @ HT('x', q2) @ HT('z', q3) @ HR('x', q4) @ HR('y', q5) @ HR('z', q6)
    R05 = T05[:3, :3]
    jw6 = R05 @ sp.Matrix([0, 0, 1])
    P6 = dk(q1, q2, q3, q4, q5, q6)[:3, 3]
    P6_joint = T05[:3, 3]
    jv6 = jw6.cross(P6 - P6_joint)
    return sp.Matrix.vstack(jv6, jw6)

# Full Jacobian matrix
def full_jacobian(q1, q2, q3, q4, q5, q6):
    J = sp.zeros(6, 6)
    J[:, 0] = J1(q1, q2, q3, q4, q5, q6)
    J[:, 1] = J2(q1, q2, q3, q4, q5, q6)
    J[:, 2] = J3(q1, q2, q3, q4, q5, q6)
    J[:, 3] = J4(q1, q2, q3, q4, q5, q6)
    J[:, 4] = J5(q1, q2, q3, q4, q5, q6)
    J[:, 5] = J6(q1, q2, q3, q4, q5, q6)
    return J

# Compute and print the Jacobian
J = full_jacobian(q1, q2, q3, q4, q5, q6)
print("Symbolic Jacobian Matrix:")
sp.pprint(J, use_unicode=True)