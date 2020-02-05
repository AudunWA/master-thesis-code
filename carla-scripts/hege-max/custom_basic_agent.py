#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
from agents.navigation.agent import AgentState
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_speed
from custom_local_planner import CustomLocalPlanner
from misc import get_distance_ahead


class CustomBasicAgent(BasicAgent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(CustomBasicAgent, self).__init__(vehicle)

        self._proximity_threshold = 15.0  # meters
        args_lateral_dict = {
            'K_P': 0.75,
            'K_D': 0.001,
            'K_I': 1,
            'dt': 1.0/20.0}
        self._local_planner = CustomLocalPlanner(
            self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict':args_lateral_dict})

    def _is_light_red_custom(self):
        # TODO: need to check decreasing distance?

        traffic_light_state = self._vehicle.get_traffic_light_state()
        if traffic_light_state == carla.TrafficLightState.Red:
            return True
        return False

    def set_target_speed(self, target_speed):
        self._local_planner.set_target_speed(target_speed)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state = self._is_light_red_custom()

        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD')

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step(debug=debug)
            close_vehicle, _ = self._is_vehicle_close(vehicle_list)

            if close_vehicle:
                control.throttle = 0

        return control

    def get_closest_vehicle_ahead(self, vehicle_list, proximity_threshold):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        ego_vehicle_yaw = self._vehicle.get_transform().rotation.yaw

        closest_distance = float('inf')
        closest_vehicle = None

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            target_vehicle_yaw = target_vehicle.get_transform().rotation.yaw

            angle_diff = ego_vehicle_yaw - target_vehicle_yaw
            angle_diff = abs((angle_diff + 180) % 360 - 180)
            if angle_diff >= 100:
                continue

            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())

            loc = target_vehicle.get_location()

            vector_angle, distance = get_distance_ahead(loc, ego_vehicle_location,
                                                        self._vehicle.get_transform().rotation.yaw,
                                                        proximity_threshold)
            if not distance:
                continue

            if target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                if vector_angle < 1 and angle_diff < 1.5:
                    # print(vector_angle, angle_diff)
                    pass
                else:
                    continue
                    # elif target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id and vector_angle<1:
            #    print(vector_angle)

            if closest_distance > distance:
                closest_distance = distance
                closest_vehicle = target_vehicle

        return (closest_vehicle, closest_distance)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        vehicle, distance = self.get_closest_vehicle_ahead(vehicle_list, max(10,get_speed(self._vehicle)/2.3))

        if(vehicle != None):
            return (True, vehicle)
        return (False, None)

    def _is_vehicle_close(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """
        vehicle, distance = self.get_closest_vehicle_ahead(vehicle_list, max(10*1.17,get_speed(self._vehicle)/2.3)*1.2)

        if(vehicle != None):
            return (True, vehicle)
        return (False, None)