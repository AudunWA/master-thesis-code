#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
from typing import List, Optional, Tuple

import numpy as np

import carla
from carla.libcarla import World


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x,
                              target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)

def compute_magnitude_angle_new(target_location, current_location, target_orientation, current_orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])

    x = math.cos(target_orientation)
    z = math.sin(-target_orientation)
    target_rotation_as_vector = np.array([x, z])

    norm_target = np.linalg.norm(target_vector)
    forward_vector = np.array([math.cos(math.radians(current_orientation)), math.sin(math.radians(current_orientation))])

    d_angle = math.degrees(math.acos(np.clip(np.dot(target_rotation_as_vector, target_vector) / (norm_target*1), -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def distance_transforms(t1, t2):
    loc = t1.location
    dx = t2.location.x - loc.x
    dy = t2.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def distance_with_dir(t1: carla.Transform, t2: carla.Transform):
    diff = t2.location - t1.location  # type: carla.Location
    distance = math.sqrt(diff.x * diff.x + diff.y * diff.y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def get_nearest_traffic_light(vehicle: carla.Vehicle) -> Tuple[
    carla.TrafficLight, float]:
    """
    This method is specialized to check European style traffic lights.

    :param lights_list: list containing TrafficLight objects
    :return: a tuple given by (bool_flag, traffic_light), where
             - bool_flag is True if there is a traffic light in RED
              affecting us and False otherwise
             - traffic_light is the object itself or None if there is no
               red traffic light affecting us
    """
    world = vehicle.get_world()  # type: World
    lights_list = world.get_actors().filter("*traffic_light*")  # type: List[carla.TrafficLight]

    ego_vehicle_location = vehicle.get_location()
    """
    map = world.get_map()
    ego_vehicle_waypoint = map.get_waypoint(ego_vehicle_location)

    closest_traffic_light = None  # type: Optional[carla.TrafficLight]
    closest_traffic_light_distance = math.inf
    for traffic_light in lights_list:
        object_waypoint = map.get_waypoint(traffic_light.get_location())
        if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        distance_to_light = distance_transforms(traffic_light.get_transform(), vehicle.get_transform())
        if distance_to_light < closest_traffic_light_distance:
            closest_traffic_light = traffic_light
            closest_traffic_light_distance = distance_to_light

    return closest_traffic_light, closest_traffic_light_distance
    """
    min_angle = 180.0
    closest_traffic_light = None  # type: Optional[carla.TrafficLight]
    closest_traffic_light_distance = math.inf
    min_rotation_diff = 0
    for traffic_light in lights_list:
        loc = traffic_light.get_location()
        distance_to_light, angle = compute_magnitude_angle(loc,
                                                   ego_vehicle_location,
                                                   vehicle.get_transform().rotation.yaw)
        rotation_diff = math.fabs(vehicle.get_transform().rotation.yaw - (traffic_light.get_transform().rotation.yaw - 90))

        if distance_to_light < closest_traffic_light_distance and angle < 90 and rotation_diff < 30:
            closest_traffic_light_distance = distance_to_light
            closest_traffic_light = traffic_light
            min_angle = angle
            min_rotation_diff = rotation_diff

    # if closest_traffic_light is not None:
    #     print("Ego rot: ", vehicle.get_transform().rotation.yaw, "TL rotation: ", closest_traffic_light.get_transform().rotation.yaw, ", diff: ", min_rotation_diff, ", dist: ", closest_traffic_light_distance)
    return closest_traffic_light, closest_traffic_light_distance
