import math
import numpy as np

def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def calculate_vector(point1, point2):
    """Calculate the vector from point1 to point2."""
    return np.array(point2) - np.array(point1)

def calculate_cos_angle(vector1, vector2):
    """Calculate the cosine of the angle between two vectors."""
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cos_angle

def quaternion_to_forward_vector(quaternion):
    # Assuming quaternion is in (x, y, z, w) format
    x, y, z, w = quaternion
    # Forward vector
    fx = 2 * (x*z + w*y)
    fy = 2 * (y*z - w*x)
    fz = 1 - 2 * (x*x + y*y)
    return np.array([fx, fy, fz])