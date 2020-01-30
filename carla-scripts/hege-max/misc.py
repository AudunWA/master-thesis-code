import math

import numpy as np


def get_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.
    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return 0

    if norm_target > max_distance:
        return False, False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))
    if d_angle < 100:
        return d_angle, norm_target
    return False, False


def get_distance(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y

    return math.sqrt(dx * dx + dy * dy)
