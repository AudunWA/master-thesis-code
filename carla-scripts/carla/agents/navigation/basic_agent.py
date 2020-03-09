#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """
import math
from typing import List

import carla
from carla.libcarla import ActorList, Actor

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import get_nearest_traffic_light, get_speed


class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

        self.stopping_for_traffic_light = False
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 0.75,
            'K_D': 0.001,
            'K_I': 1,
            'dt': 1.0 / 20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None
        self.drawn_lights = False
        self.is_affected_by_traffic_light = False

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """


        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()  # type: ActorList
        vehicle_list = actor_list.filter("*vehicle*")  # type: List[Actor]
        pedestrians_list = actor_list.filter("*walker.pedestrian*")
        lights_list = actor_list.filter("*traffic_light*")  # type: List[carla.TrafficLight]

        if not self.drawn_lights and debug:
            for light in lights_list:
                self._world.debug.draw_box(
                    carla.BoundingBox(light.trigger_volume.location + light.get_transform().location,
                                      light.trigger_volume.extent * 2),
                    carla.Rotation(0, 0, 0), 0.05, carla.Color(255, 128, 0, 0), 0)
            self.drawn_lights = True

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # Check for pedestrians
        pedestrian_state, pedestrian = self._is_pedestrian_hazard(pedestrians_list)
        if pedestrian_state:
            if debug:
                print('!!! PEDESTRIAN BLOCKING AHEAD [{}])'.format(pedestrian.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        new_target_speed = self._update_target_speed(hazard_detected, debug)

        # if hazard_detected:
        #     control = self.emergency_stop()
        # else:
        #     self._state = AgentState.NAVIGATING
        #     self.braking_intial_speed = None
        #     # standard local planner behavior
        #     control = self._local_planner.run_step(debug=debug)
        #     if self.stopping_for_traffic_light:
        #         control.steer = 0.0

        self._state = AgentState.NAVIGATING
        self.braking_intial_speed = None
        # standard local planner behavior
        control = self._local_planner.run_step(debug=debug)
        if self.stopping_for_traffic_light:
            control.steer = 0.0
        # Prevent from steering randomly when stopped
        if math.fabs(get_speed(self._vehicle)) < 0.1:
            control.steer = 0

        return control

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        return self._local_planner.done()

    def _update_target_speed(self, hazard_detected, debug):
        if hazard_detected:
            self._set_target_speed(0)
            return 0

        MAX_PERCENTAGE_OF_SPEED_LIMIT = 0.75
        speed_limit = self._vehicle.get_speed_limit()  # km/h
        current_speed = get_speed(self._vehicle)
        new_target_speed = speed_limit * MAX_PERCENTAGE_OF_SPEED_LIMIT

        use_custom_traffic_light_speed = False
        if use_custom_traffic_light_speed:
            TRAFFIC_LIGHT_SECONDS_AWAY = 3
            METERS_TO_STOP_BEFORE_TRAFFIC_LIGHT = 8
            get_traffic_light = self._vehicle.get_traffic_light()  # type: carla.TrafficLight
            nearest_traffic_light, distance = get_nearest_traffic_light(self._vehicle)  # type: carla.TrafficLight, float
            distance_to_light = distance
            distance -= METERS_TO_STOP_BEFORE_TRAFFIC_LIGHT

            if nearest_traffic_light is None:
                nearest_traffic_light = get_traffic_light

            # Draw debug info
            if debug and nearest_traffic_light is not None:
                self._world.debug.draw_point(
                    nearest_traffic_light.get_transform().location,
                    size=1,
                    life_time=0.1,
                    color=carla.Color(255, 15, 15))
            """
            if get_traffic_light is not None:
                print("get_traffic_light:     ", get_traffic_light.get_location() if get_traffic_light is not None else "None", " ", get_traffic_light.state if get_traffic_light is not None else "None")
    
            if nearest_traffic_light is not None:
                print("nearest_traffic_light: ",  nearest_traffic_light.get_location() if nearest_traffic_light is not None else "None", " ", nearest_traffic_light.state if nearest_traffic_light is not None else "None")
            """

            ego_vehicle_location = self._vehicle.get_location()
            ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

            self.is_affected_by_traffic_light = False
            self.stopping_for_traffic_light = False
            if ego_vehicle_waypoint.is_junction:
                # It is too late. Do not block the intersection! Keep going!
                pass

            # Check if we should start braking
            elif distance_to_light <= TRAFFIC_LIGHT_SECONDS_AWAY * new_target_speed / 3.6 and nearest_traffic_light is not None and nearest_traffic_light.state != carla.TrafficLightState.Green:
                self.is_affected_by_traffic_light = True
                brake_distance = current_speed / 3.6 * TRAFFIC_LIGHT_SECONDS_AWAY
                print("TL distance: ", distance_to_light, ", distance (to stop): ", distance, ", distance travel 4 secs: ", brake_distance)
                new_target_speed = self._target_speed
                if distance <= 0:
                    new_target_speed = 0
                    self.stopping_for_traffic_light = True
                    print("Stopping before traffic light, distance  ", distance, "m")
                elif brake_distance >= distance and brake_distance != 0:
                    percent_before_light = (brake_distance - distance) / brake_distance
                    new_target_speed = speed_limit - max(0, percent_before_light) * speed_limit
                    print("Slowing down before traffic light ", percent_before_light * 100, "% ", new_target_speed, " km/h")

        self._set_target_speed(max(0, new_target_speed))
        return new_target_speed

    def _set_target_speed(self, target_speed: int):
        """
        This function updates all the needed values required to actually set a new target speed
        """
        self._target_speed = target_speed
        self._local_planner.set_speed(target_speed)
